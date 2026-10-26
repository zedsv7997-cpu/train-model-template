"""Tỉa bảng từ vựng Qwen3.5 (248K token) chỉ còn tiếng Việt, tiếng Anh và token của dữ liệu train.

Giữ lại:
  1. 256 token byte  -> chữ nào cũng vẫn mã hóa được (ký tự lạ tách thành byte), không bao giờ "mất chữ"
  2. token đặc biệt (<|im_start|>, <tool_call>, <think>...)
  3. MỌI token xuất hiện trong prompt + tool + hội thoại của dữ liệu (--keep-texts), sau khi render bằng chat template
  4. token phổ biến trong Wikipedia tiếng Việt + tiếng Anh (--counts = data/token_counts.json, đếm sẵn), CHỈ khi mọi ký tự là
     chữ Latin / dấu tiếng Việt / số / dấu câu / ký hiệu — chữ Trung, Nhật, Hàn, Nga, Ả Rập... bị xóa
  (+ mọi token "cha" trong chuỗi merge BPE của các token trên, để tokenizer vẫn ghép ra được chúng)
Phần còn lại bị xóa: dòng tương ứng trong embedding (dùng chung với lm_head), merges, id trong config.

  python scripts/prune_vocab.py --model outputs/qwen35-2b-lora/merged --out outputs/qwen35-2b-lora/merged-vi-en
      --counts <token_counts.json> --total 65536
"""
import argparse
import json
import os
import shutil
import sys
import time
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from chat_tokens import encode, render, token_ids

sys.stdout.reconfigure(encoding="utf-8")

ALLOWED = [  # khối Unicode được coi là "tiếng Việt / tiếng Anh / ký hiệu"
    (0x0000, 0x024F),   # ASCII, Latin-1, Latin Extended-A/B (đ, ơ, ư...)
    (0x0300, 0x036F),   # dấu kết hợp
    (0x1E00, 0x1EFF),   # Latin Extended Additional (ạ, ế, ữ...)
    (0x2000, 0x22FF),   # dấu câu, tiền tệ (₫), mũi tên (→), ký hiệu toán
    (0x2460, 0x25FF),   # số khoanh tròn, khung, hình học
    (0x2600, 0x27BF),   # ký hiệu linh tinh
    (0xFE0F, 0xFE0F),   # variation selector của emoji
    (0x1F300, 0x1FAFF),  # emoji (hay gặp trong câu trả lời của chatbot)
]


def byte_decoder():
    # bảng byte <-> unicode của BPE byte-level (GPT-2)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DEC = byte_decoder()


def is_vi_en(token):
    try:
        raw = bytes(BYTE_DEC[ch] for ch in token)
    except KeyError:
        return False
    text = raw.decode("utf-8", errors="ignore")  # mảnh byte dở dang -> bỏ phần dở, xét phần đọc được
    return all(any(lo <= ord(ch) <= hi for lo, hi in ALLOWED) for ch in text)


def split_merge(m):
    if isinstance(m, str):
        a, b = m.split(" ", 1)
        return a, b
    return m[0], m[1]


def remap_token_ids(obj, old2new):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.endswith("token_id") or k.endswith("token_ids"):
                if isinstance(v, int) and v in old2new:
                    obj[k] = old2new[v]
                elif isinstance(v, list):
                    obj[k] = [old2new.get(x, x) if isinstance(x, int) else x for x in v]
            else:
                remap_token_ids(v, old2new)
    elif isinstance(obj, list):
        for v in obj:
            remap_token_ids(v, old2new)


def domain_texts(paths, tok, limit=0):
    """Render hội thoại đúng như lúc chạy (system + tools + chat template) để lấy đủ token. limit > 0: chỉ N hội thoại
    đầu mỗi file .jsonl."""
    for p in paths:
        if p.endswith(".jsonl"):
            with open(p, encoding="utf-8") as f:
                n = 0
                for line in f:
                    if line.strip():
                        yield render(tok, json.loads(line))
                        n += 1
                        if limit and n >= limit:
                            break
        else:
            with open(p, encoding="utf-8") as f:
                yield f.read()


def compute_keep(src, counts_path, total_tokens, min_count, keep_texts):
    """Chọn tập token giữ lại (các bước 1-4 ở đầu file) cho model ở thư mục src (chỉ cần tokenizer + config).
    Tất định theo đầu vào, nên train.py (--vocab-keep) gọi được TRƯỚC khi train để tính loss trên đúng từ vựng
    mà prune sẽ giữ sau đó. Trả về namespace: new_rows (id cũ theo thứ tự id mới), old2new, special_above..."""
    with open(os.path.join(src, "tokenizer.json"), encoding="utf-8") as f:
        tj = json.load(f)
    with open(os.path.join(src, "tokenizer_config.json"), encoding="utf-8") as f:
        tc = json.load(f)
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    vocab, merges = tj["model"]["vocab"], tj["model"]["merges"]
    inv = {i: t for t, i in vocab.items()}
    n_base = len(vocab)
    old_vocab_size = (cfg.get("text_config") or cfg).get("vocab_size")
    atd = tc.get("added_tokens_decoder", {})
    special_ids = sorted({t["id"] for t in tj["added_tokens"]} | {int(k) for k in atd})
    special_above = [i for i in special_ids if i >= n_base]
    forced_base = [i for i in special_ids if i < n_base]
    budget = total_tokens - len(special_above)

    producer = {}
    for m in merges:
        a, b = split_merge(m)
        producer.setdefault(a + b, (a, b))
    byte_tokens = [t for t in vocab if len(t) == 1]
    assert len(byte_tokens) == 256

    keep = set(byte_tokens) | {inv[i] for i in forced_base}

    def ancestors(t0):
        need, stack = [], [t0]
        while stack:
            t = stack.pop()
            if t in keep or t in need:
                continue
            need.append(t)
            if t in producer:
                stack.extend(producer[t])
        return need

    # 3. token của dữ liệu: bắt buộc giữ
    old_tok = AutoTokenizer.from_pretrained(src)
    sm_ids, n_texts = token_ids(old_tok, domain_texts(keep_texts, old_tok))
    for i in sorted(sm_ids):
        if i < n_base:
            keep.update(ancestors(inv[i]))
    print(f"[data] {n_texts} văn bản, {len(sm_ids)} token khác nhau -> giữ {len(keep)} token (gồm byte + đặc biệt)")
    n_forced = len(keep)

    # 4. token tiếng Việt / tiếng Anh theo tần suất, lọc theo chữ viết
    with open(counts_path, encoding="utf-8") as f:
        counts = json.load(f)
    order = sorted(((int(i), c) for i, c in counts.items() if int(i) < n_base), key=lambda x: -x[1])
    total_occ = sum(c for _, c in order)
    covered = dropped_script = 0
    for i, c in order:
        if c < min_count or len(keep) >= budget:
            break
        t = inv[i]
        if t in keep:
            covered += c
            continue
        if not is_vi_en(t):
            dropped_script += 1
            continue
        need = ancestors(t)
        if len(keep) + len(need) > budget:
            continue
        keep.update(need)
        covered += c
    n_vi_en_seen = sum(1 for i, c in order if c >= min_count)
    print(f"[vi+en] thêm {len(keep) - n_forced} token phổ biến (phủ {100 * covered / total_occ:.3f}% văn bản Wikipedia vi+en), "
          f"bỏ {dropped_script} token khác chữ viết; {n_vi_en_seen} token có >= {min_count} lần xuất hiện")

    kept_base_ids = sorted(vocab[t] for t in keep)
    old2new = {old: new for new, old in enumerate(kept_base_ids)}
    for j, old in enumerate(special_above):
        old2new[old] = len(kept_base_ids) + j
    new_rows = kept_base_ids + special_above
    total = len(new_rows)
    print(f"[kết quả] {old_vocab_size} -> {total} token ({len(kept_base_ids)} thường + {len(special_above)} đặc biệt)")
    return SimpleNamespace(tj=tj, tc=tc, cfg=cfg, vocab=vocab, merges=merges, inv=inv, old_vocab_size=old_vocab_size,
                           atd=atd, special_above=special_above, keep=keep, kept_base_ids=kept_base_ids,
                           old2new=old2new, new_rows=new_rows, total=total, old_tok=old_tok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="thư mục model HF (bản merge sau fine-tune)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--counts", default="data/token_counts.json", help="token_counts.json (đếm trên Wikipedia vi + en)")
    ap.add_argument("--total", type=int, default=65536, help="trần số token sau khi tỉa (kể cả token đặc biệt)")
    ap.add_argument("--min-count", type=int, default=2)
    # mọi văn bản model sẽ gặp: dữ liệu train + test + calib của imatrix (.jsonl render qua chat template, file khác đọc thô)
    ap.add_argument("--keep-texts", nargs="+", default=[
        p for p in ("data/train.jsonl", "data/test.jsonl", "data/calib.txt") if os.path.exists(p)])
    ap.add_argument("--check-n", type=int, default=500,
                    help="số hội thoại đầu mỗi file .jsonl trong --keep-texts dùng để kiểm tra tách token y hệt bản cũ")
    args = ap.parse_args()
    src, out = args.model, args.out
    os.makedirs(out, exist_ok=True)
    k = compute_keep(src, args.counts, args.total, args.min_count, args.keep_texts)
    tj, tc, cfg, merges, inv, old_vocab_size, atd = k.tj, k.tc, k.cfg, k.merges, k.inv, k.old_vocab_size, k.atd
    keep, kept_base_ids, old2new, new_rows, total, old_tok = (
        k.keep, k.kept_base_ids, k.old2new, k.new_rows, k.total, k.old_tok)

    # tokenizer
    new_vocab = {inv[old]: old2new[old] for old in kept_base_ids}
    new_merges = [m for m in merges if all(x in keep for x in (*split_merge(m), "".join(split_merge(m))))]
    tj["model"]["vocab"], tj["model"]["merges"] = new_vocab, new_merges
    for t in tj["added_tokens"]:
        t["id"] = old2new[t["id"]]
    tj["added_tokens"].sort(key=lambda t: t["id"])
    with open(os.path.join(out, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tj, f, ensure_ascii=False)
    tc["added_tokens_decoder"] = {str(old2new[int(k)]): v for k, v in atd.items()}
    with open(os.path.join(out, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tc, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(new_vocab, f, ensure_ascii=False)
    with open(os.path.join(out, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for m in new_merges:
            f.write("%s %s\n" % split_merge(m))
    print(f"[tokenizer] merges {len(merges)} -> {len(new_merges)}")

    # config + các file phụ
    remap_token_ids(cfg, old2new)
    for c in (cfg, cfg.get("text_config", {})):
        if "vocab_size" in c:
            c["vocab_size"] = total
    cfg["_vocab_pruned_from"] = {"model": src, "old_vocab_size": old_vocab_size, "kept": total, "langs": "vi,en,data"}
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    written = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"}
    for name in os.listdir(src):
        p = os.path.join(src, name)
        if name in written or name.endswith(".safetensors") or name.endswith(".index.json") or not os.path.isfile(p):
            continue
        if name == "generation_config.json":
            with open(p, encoding="utf-8") as f:
                g = json.load(f)
            remap_token_ids(g, old2new)
            with open(os.path.join(out, name), "w", encoding="utf-8") as f:
                json.dump(g, f, ensure_ascii=False, indent=2)
        else:
            shutil.copy(p, os.path.join(out, name))

    # trọng số: cắt mọi chiều có kích thước = vocab cũ (embed_tokens, lm_head nếu có)
    index = torch.tensor(new_rows, dtype=torch.long)
    shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    weight_map, total_bytes, t0 = {}, 0, time.perf_counter()
    for shard in shards:
        sd = load_file(os.path.join(src, shard))
        touched = []
        for name, w in sd.items():
            dims = [d for d, s in enumerate(w.shape) if s == old_vocab_size]
            for d in dims:
                w = w.index_select(d, index)
            if dims:
                sd[name] = w.contiguous()
                touched.append(f"{name} {tuple(sd[name].shape)}")
        save_file(sd, os.path.join(out, shard), metadata={"format": "pt"})
        for name, w in sd.items():
            weight_map[name] = shard
            total_bytes += w.numel() * w.element_size()
        print(f"[weights] {shard}: cắt {touched}")
        del sd
    print(f"[weights] xong trong {time.perf_counter() - t0:.0f}s, tổng {total_bytes / 1e9:.2f} GB")

    # kiểm tra: token đặc biệt remap đúng, hội thoại tách token GIỐNG HỆT bản cũ, decode khớp
    new_tok = AutoTokenizer.from_pretrained(out)
    for s in ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "<|endoftext|>"):
        o, n = old_tok.convert_tokens_to_ids(s), new_tok.convert_tokens_to_ids(s)
        assert old2new[o] == n, (s, o, n)
    # --check-n hội thoại đầu mỗi file .jsonl trong --keep-texts (cả bộ train lớn thì tách token chậm)
    texts = list(domain_texts([p for p in args.keep_texts if p.endswith(".jsonl")], old_tok, args.check_n))
    diff = sum([old2new.get(i, -1) for i in o_ids] != n_ids
               for o_ids, n_ids in zip(encode(old_tok, texts), encode(new_tok, texts)))
    print(f"[check] token đặc biệt remap đúng; hội thoại tách token khác bản cũ ở {diff}/{len(texts)} văn bản")
    assert diff == 0, "từ vựng đã tỉa tách prompt khác bản gốc -> model thấy token khác lúc train"
    for s in ["Hôm nay thời tiết Hà Nội thế nào?", "Book a meeting room for tomorrow at 9am",
              "Giá: 1.250.000 ₫ — 25°C → km²", "日本語 русский 한국어"]:
        o, n = old_tok(s, add_special_tokens=False)["input_ids"], new_tok(s, add_special_tokens=False)["input_ids"]
        assert new_tok.decode(n) == s
        print(f"[check] {len(o):3d} -> {len(n):3d} token: {s}")

    # Ghi SAU CÙNG, khi mọi kiểm tra đã qua: có file này nghĩa là tỉa xong
    with open(os.path.join(out, "vocab_old2new.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in old2new.items()}, f)
    if len(shards) > 1 or os.path.exists(os.path.join(src, "model.safetensors.index.json")):
        with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
            json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
