from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import time
from pathlib import Path
import json
from tqdm import tqdm

from llama2.model import ModelArgs, Transformer
from sentencepiece import SentencePieceProcessor
class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args

    @staticmethod
    def build(ckpt_dir: str, tokenizer_path: str, load_model: bool = True, max_seq_len: int = 2048, max_batch_size: int = 32, device: str = "cuda"):
        prev_time = time.time()
        if load_model:
            ckpt = sorted(Path(ckpt_dir).glob("*.pth"))
            assert len(ckpt) > 0, f"No checkpoint found in {ckpt_dir}"
            ckpt_path = ckpt[0]
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f} seconds")
            prev_time = time.time()

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded model in {time.time() - prev_time:.2f} seconds")
        
        return LLaMA(model, tokenizer, model_args)
