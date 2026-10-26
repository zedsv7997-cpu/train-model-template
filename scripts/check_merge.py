"""Kiểm tra bản merge 16-bit: mỗi tensor có LoRA phải bằng gốc + (alpha/r)·B@A, tensor khác phải y hệt gốc, không NaN/Inf.

Từng gặp (2026-10-19): LoRA đúng nhưng bản merge của Unsloth có 5 phần tử rác (tới 5.7e20, như byte float32 đọc
thành bf16) ở layers.1.mlp.down_proj -> model GGUF sinh toàn ký tự vô nghĩa. Chạy sau bước train để bắt lỗi này.

  %USERPROFILE%\\ml-env\\Scripts\\python.exe scripts\\check_merge.py outputs\\qwen35-2b-lora
Thoát mã 1 nếu có sai lệch; in các tensor lệch nhiều nhất.
"""
import argparse
import glob
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def local_base(name):
    if os.path.isdir(name):
        return name
    from huggingface_hub import snapshot_download
    return snapshot_download(name, local_files_only=True)


def open_dir(d):
    handles = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        h = safe_open(f, "pt")
        for k in h.keys():
            handles[k] = h
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="outputs\\qwen35-2b-<Tag> (có lora\\ và merged\\)")
    ap.add_argument("--tol", type=float, default=0.02, help="sai lệch tuyệt đối tối đa (làm tròn bf16 ~1e-3)")
    # LoRA chỉ đổi trọng số cỡ 1e-3 (đo: tối đa 0.0023), nhỏ hơn --tol -> tensor chưa được merge (merge chết giữa chừng)
    # vẫn lọt ngưỡng tuyệt đối. Đo: bản merge đúng lệch ‖t-want‖ <= 0.25·‖LoRA‖ (làm tròn bf16), chưa merge = 1.0
    ap.add_argument("--rel-tol", type=float, default=0.5, help="‖merge - (gốc+LoRA)‖ tối đa, tính theo ‖LoRA‖")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = json.load(open(os.path.join(args.out, "lora", "adapter_config.json"), encoding="utf-8"))
    lora = load_file(os.path.join(args.out, "lora", "adapter_model.safetensors"))
    scale = cfg["lora_alpha"] / cfg["r"]
    base = open_dir(local_base(cfg["base_model_name_or_path"]))
    merged = open_dir(os.path.join(args.out, "merged"))

    bad, n_lora, n_same = [], 0, 0
    bad += [(float("inf"), f"{k} (thiếu trong bản merge)") for k in base if k not in merged]
    for k, h in merged.items():
        t = h.get_tensor(k)
        if t.is_floating_point() and not torch.isfinite(t).all():
            bad.append((float("inf"), k))
            continue
        if k not in base:
            continue
        w = base[k].get_tensor(k)
        lk = "base_model.model." + k[: -len(".weight")] if k.endswith(".weight") else None
        a = lora.get(f"{lk}.lora_A.weight") if lk else None
        delta = None
        if a is not None:
            delta = scale * (lora[f"{lk}.lora_B.weight"].float() @ a.float())
            want = w.float() + delta
            n_lora += 1
        elif t.shape == w.shape:
            want = w.float()
            n_same += 1
        else:
            continue
        diff = t.float() - want
        err = diff.abs().max().item()
        rel = (diff.norm() / delta.norm()).item() if delta is not None else 0.0
        if err > args.tol:
            bad.append((err, k))
        elif rel > args.rel_tol:
            bad.append((err, f"{k} (lệch {rel:.2f}·‖LoRA‖ - chưa merge?)"))
    print(f"check_merge {args.out}: {n_lora} tensor có LoRA, {n_same} tensor giữ nguyên, {len(bad)} tensor sai")
    for err, k in sorted(bad, reverse=True)[:10]:
        print(f"  SAI {err:.3g}  {k}")
    if n_lora == 0:
        print("  SAI: không tìm thấy tensor nào có LoRA (tên khóa khác?)")
        return 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
