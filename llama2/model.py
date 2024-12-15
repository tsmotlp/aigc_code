import torch
import torch.nn as nn

import torch.nn.functional as F

import math
import dataclasses as dataclass

from typing import Optional, List, Tuple, Dict, Any


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256 
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = "cuda" if torch.cuda.is_available() else "cpu"    


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As write in the paper, the dimension of the head must be even
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Build the theta parameter
    # According to the formula: theta_i = 10000 ^ (-2(i-1)/(dim)) for i in [1, 2, ..., dim/2]
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Consruct the positions (the "m" parameter)
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)

    # Multiply each theta by each position using the outer product
    # Shape: (seq_len) outer product *(head_dim/2) -> (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).to(device)

    # We can compute complex numbers in the polar form c = R * e^(i*m*theta), where R = 1 and theta is the theta parameter
    # We can then compute the complex numbers in the cartesian form c = a + bi, where a = R * cos(m*theta) and b = R * sin(m*theta)
    # Shape: (seq_len, head_dim/2) -> (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tesnsor, device: str):
    # x shape: [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, num_heads, head_dim/2]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_complex shape: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (batch_size, seq_len, num_heads, head_dim/2) * (1, seq_len, 1, head_dim/2) = (batch_size, seq_len, num_heads, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (batch_size, seq_len, num_heads, head_dim/2) -> (batch_size, seq_len, num_heads, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (batch_size, seq_len, num_heads, head_dim/2, 2) -> (batch_size, seq_len, num_heads, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # x shape: [batch_size, seq_len, dim]
        return self.weight * self._norm(x.float()).type_as(x)

def repeat_kv(kv: torch.Tensor, n_rep: int):
    # kv shape: (batch_size, seq_len, n_kv_heads, head_dim)
    # n_rep: number of repetitions
    # Shape: (batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    batch_size, seq_len, n_kv_heads, head_dim = kv.shape
    if n_rep == 1:
        return kv
    else:
        # Shape: (batch_size, seq_len, n_kv_heads, head_dim) -> (batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        return kv[:,:,:,None,:].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim) 
    
class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.n_heads_q = args.n_heads
        # Indicates how many times the key and value are repeated to match the head dimension of queries
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Number of dimensions of each head
        self.head_dim = args.dim // self.n_heads
        
        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads_q * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)).to(args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)).to(args.device)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape    # x shape: [batch_size, 1, dim]
        # Apply the linear transformation to the input
        xq = self.wq(x) # xq shape: [batch_size, 1, dim] -> [batch_size, 1, n_heads_q * head_dim] 
        xk = self.wk(x) # xk shape: [batch_size, 1, n_kv_heads * head_dim]
        xv = self.wv(x) # xv shape: [batch_size, 1, n_kv_heads * head_dim]

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)    # xq shape: [batch_size, 1, n_heads_q, head_dim]
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)    # xk shape: [batch_size, 1, n_kv_heads, head_dim]
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)    # xv shape: [batch_size, 1, n_kv_heads, head_dim]

        # Apply the rotary embeddings
        xq = apply_rotary_embeddings(xq, freqs_complex, x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, x.device)

        # Cache the keys and values
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        # Retrieve all the cached keys and values so far
        # Shape: (batch_size, seq_len, n_kv_heads, head_dim)
        keys = self.cache_k[:batch_size, :start_pos+seq_len]
        values = self.cache_v[:batch_size, :start_pos+seq_len]   

        # Repeat the heads of the keys and values to reach the number of heads of the queries
        # Shape: (batch_size, seq_len, n_heads_q, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2) # Shape: (batch_size, n_heads_q, seq_len, head_dim)
        keys = keys.transpose(1, 2) # Shape: (batch_size, n_kv_heads, seq_len, head_dim)
        values = values.transpose(1, 2) # Shape: (batch_size, n_kv_heads, seq_len, head_dim)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # Shape: (batch_size, n_heads_q, seq_len, seq_len)

        # Apply the attention scores to the values
        output = torch.matmul(scores, values) # Shape: (batch_size, n_heads_q, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # Shape: (batch_size, seq_len, dim)

        return self.wo(output)  # Shape: (batch_size, seq_len, dim)
    
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        hidden_dim = args.dim * 4
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)
        
        # Round the hidden dimension to the nearest multiple of the multiple_of parameter
        hidden_dim = (hidden_dim + args.multiple_of - 1) // args.multiple_of * args.multiple_of

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads

        self.self_attn = SelfAttention(args)
        self.ffn = FeedForward(args)

        # Normalization beforer the attention and feed-forward layers   
        self.attention_norm = RMSNorm(self.dim, eps=self.args.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, eps=self.args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # x shape: [batch_size, seq_len, dim]
        h = x + self.self_attn(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.ffn(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size != -1, "Vocab size must be set"
        self.args = args

        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.output = nn.Linear(self.args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # tokens: [batch_size, seq_len]
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"
        # h: [batch_size, seq_len, dim]
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the current position [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apply the transformer layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        # Apply the final layer norm
        h = self.norm(h)

        # Output the last token
        output = self.output(h).float()

        return output



