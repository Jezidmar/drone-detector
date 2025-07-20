import torch
from pathlib import Path
import argparse

def adapt_keys(key: str):
    if "modality_encoders" in key:
        chunks = key.split(".")
        if "decoder" in key or "context_encoder" in key:
            return None
        return ".".join(chunks[2:])          # drop first two tokens
    if "_ema" in key:
        return None
    return key

def convert(src: str, dst: str):
    sd = torch.load(src, map_location="cpu")

    new_sd = {new_k: w for k, w in sd["model"].items()
                       if (new_k := adapt_keys(k)) is not None}

    torch.save(new_sd, dst)
    print(f"Patched {len(new_sd)} tensors\n→ {dst}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="original checkpoint")
    p.add_argument("--dst", default="patched.pth", help="output path")
    args = p.parse_args()

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    convert(args.src, args.dst)