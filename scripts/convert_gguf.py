"""Gọi convert_hf_to_gguf.py của llama.cpp, chấp nhận được model đã tỉa từ vựng (prune_vocab.py).

llama.cpp nhận diện pre-tokenizer bằng mã băm của kết quả tách một chuỗi mẫu. Tỉa từ vựng làm đổi id token nên mã
băm đổi theo và convert báo "BPE pre-tokenizer was not recognized", dù quy tắc tách chữ vẫn y nguyên của Qwen3.5.
Wrapper này chỉ can thiệp đúng trường hợp đó: model có dấu "_vocab_pruned_from" trong config thì dùng "qwen35".

  python scripts/convert_gguf.py <thư mục model HF> --outtype bf16 --outfile <file.gguf>
"""
import json
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "tools", "llama.cpp-src")
sys.path.insert(0, SRC)
sys.path.insert(1, os.path.join(SRC, "gguf-py"))

import conversion.base as base  # noqa: E402

model_dir = next((a for a in sys.argv[1:] if os.path.isdir(a)), None)
pruned = False
if model_dir and os.path.exists(os.path.join(model_dir, "config.json")):
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        pruned = "_vocab_pruned_from" in json.load(f)

_orig = base.TextModel.get_vocab_base_pre


def get_vocab_base_pre(self, tokenizer):
    try:
        return _orig(self, tokenizer)
    except NotImplementedError:
        if pruned:
            print("convert_gguf: model đã tỉa từ vựng -> dùng pre-tokenizer 'qwen35'")
            return "qwen35"
        raise


base.TextModel.get_vocab_base_pre = get_vocab_base_pre
sys.argv[0] = os.path.join(SRC, "convert_hf_to_gguf.py")
runpy.run_path(sys.argv[0], run_name="__main__")
