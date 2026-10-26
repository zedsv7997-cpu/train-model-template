"""Tạo data/calib.txt cho llama-imatrix (quantize.ps1): N hội thoại ngẫu nhiên của bộ train, render bằng chat template
đúng như lúc chạy (system + tools + các lượt), mỗi hội thoại một khối.

  %USERPROFILE%\\ml-env\\Scripts\\python.exe scripts\\make_calib.py --train data\\train.jsonl --n 40
"""
import argparse
import json
import random

from transformers import AutoTokenizer

from chat_tokens import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B", help="lấy tokenizer / chat template")
    ap.add_argument("--out", default="data/calib.txt")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    with open(args.train, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in random.Random(args.seed).sample(rows, min(args.n, len(rows))):
            f.write(render(tok, ex) + "\n")
    print(f"calib: {args.out} ({min(args.n, len(rows))} hội thoại)")


if __name__ == "__main__":
    main()
