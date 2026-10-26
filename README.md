# Train Qwen3.5 gọi tool trên Windows (RTX 8 GB) rồi nén bằng llama.cpp

Khung fine-tune **Qwen3.5-2B / 0.8B** gọi tool bằng Unsloth bf16 LoRA trên một GPU 8 GB, rồi tỉa từ vựng (vi + en) và
nén GGUF (IQ4_XS) để chạy với llama-server. Chạy thẳng trên Windows, không cần CUDA Toolkit hay MSVC.

## Cài môi trường

1. **Python 3.12.10**: bản Windows LTSC không có winget / Store và trình cài `.exe` có thể lỗi MSI (0x80070003), nên
   dùng gói NuGet chính thức của python.org: tải `https://www.nuget.org/api/v2/package/python/3.12.10`, giải nén, chép
   thư mục `tools\` vào `%LOCALAPPDATA%\Programs\Python\Python312` rồi thêm thư mục đó và `\Scripts` vào PATH của user.
2. **venv** (các script gọi đúng đường dẫn này):
   ```
   python -m venv %USERPROFILE%\ml-env
   %USERPROFILE%\ml-env\Scripts\python.exe -m pip install -r setup\pip-freeze-20261026-reinstall.txt --extra-index-url https://download.pytorch.org/whl/cu130
   ```
   Bản chính: torch 2.14.1+cu130, triton-windows 3.8.0.post29, unsloth / unsloth_zoo 2026.10.7, transformers 5.17.0,
   TRL 1.15.0, PEFT 0.21.2, torchao 0.18.0, fla 0.5.2, xformers 0.0.35. Unsloth 2026.10.7 chặn transformers ≤ 5.17.0,
   TRL ≤ 1.15.0, datasets < 5.
3. **Model gốc** (4,3 GiB vào cache HF; các script chạy `HF_HUB_OFFLINE=1`):
   `python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-2B')"`
4. **llama.cpp** vào `tools\` (b11803 CUDA 13 + mã nguồn cùng bản để có `convert_hf_to_gguf.py`; không kèm trong repo):
   `setup\get_llamacpp.ps1`, rồi `setup\get_llamacpp_src.ps1`. Model gốc dạng GGUF để so sánh: `setup\get_base_models.ps1`.

## Dữ liệu

`data\train.jsonl`, mỗi dòng một hội thoại kiểu OpenAI:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
              {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                                                                   "function": {"name": "...", "arguments": {...}}}]},
              {"role": "tool", "tool_call_id": "call_1", "name": "...", "content": "..."},
              {"role": "assistant", "content": "..."}],
 "tools": [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]}
```
- Loss chỉ tính ở phần assistant. Hội thoại cùng system prompt + danh sách tool được gom nhóm 16 để phần đầu chung
  chỉ tính một lần mỗi bước (`shared_prefix.py`) — nhanh nhất khi phần đầu đó giống hệt nhau giữa các mẫu.
- Tối đa 10.752 token / hội thoại (`--max-seq-len`); dài hơn thì bỏ dần lượt cuối.
- `data\test.jsonl` (tuỳ chọn, cùng định dạng): train.py thử sinh 3 câu sau khi train.
- `data\token_counts.json`: tần suất token trên Wikipedia vi + en, dùng cho `prune_vocab.py`.

## Các bước

```
# 1. train (+ merge 16-bit vào outputs\<x>\merged); tự chạy lại từ checkpoint nếu Python crash
powershell -ExecutionPolicy Bypass -File scripts\run_train.ps1 -Out outputs\qwen35-2b-x -Log outputs\train-x -TrainArgs "--train data\train.jsonl --no-samples"
# chạy thử nhanh: %USERPROFILE%\ml-env\Scripts\python.exe scripts\train.py --limit 400 --max-steps 8 --no-merge --no-samples

# 2. kiểm tra bản merge (Unsloth từng ghi rác vào vài phần tử)
%USERPROFILE%\ml-env\Scripts\python.exe scripts\check_merge.py outputs\qwen35-2b-x

# 3. tỉa từ vựng 248K -> 65.536 token (giữ byte, token đặc biệt, mọi token của data\*.jsonl, vi + en)
%USERPROFILE%\ml-env\Scripts\python.exe scripts\prune_vocab.py --model outputs\qwen35-2b-x\merged --out outputs\qwen35-2b-x\merged-vi-en

# 4. calib cho imatrix, rồi GGUF BF16 -> imatrix -> IQ4_XS
%USERPROFILE%\ml-env\Scripts\python.exe scripts\make_calib.py
powershell -ExecutionPolicy Bypass -File scripts\quantize.ps1 -Merged outputs\qwen35-2b-x\merged-vi-en -Out models\qwen35-2b-x-vi-en

# 5. chạy: API OpenAI-compatible (tools / tool_calls) tại http://127.0.0.1:8090/v1, tự dò -ngl theo giới hạn VRAM
powershell -ExecutionPolicy Bypass -File scripts\serve.ps1 -Model models\qwen35-2b-x-vi-en-IQ4_XS.gguf
```

Script kiểm tra: `check_shared_prefix.py` (gradient prefix dùng chung khớp cách train từng hội thoại), 
`check_shared_prefix_fp32.py` (cùng phép thử trên model tí hon fp32), `ram_test.py` (thử RAM có ổn định không ngay
trong Windows, khi train crash bất thường).

## Lưu ý trên laptop (đo trên Lenovo 83M0, RTX 5060 Laptop 8 GB, Ryzen 7 H 255)

- **Bật chế độ Performance** (Fn+Q tới khi đèn nút nguồn đỏ). Balanced khóa GPU ở 75 W: chậm ~20%. Performance 115 W:
  ~9,0–9,4 s/bước (16 hội thoại, ~142K token / bước).
- **VRAM khi train khóa ~6 GB**: `TRAIN_VRAM_GB` (mặc định 6,2) là trần PyTorch; `SHARED_PREFIX_PEAK_GB` (5,0) là đỉnh
  nhắm để tự chọn số lớp không checkpoint; `train.py` bật lại `expandable_segments` mà Unsloth gỡ trên Windows (tắt:
  `TRAIN_EXPANDABLE_SEGMENTS=0`). Hội thoại 10.752 token cần ~5,7 GiB; trần 5,8 thì OOM.
- Tổng VRAM vượt 8 GB thì Windows (WDDM) âm thầm đẩy sang RAM, train chậm 5–8 lần: tắt `llama-server` khi train.
  `run_train.ps1` chờ GPU dùng ≤ 1300 MiB (`-GpuFreeMiB`; màn hình chạy trên RTX giữ sẵn ~1 GB) rồi mới train;
  `train.py` (`SlowGuard`) tự dừng + lưu checkpoint khi bước chậm bất thường.
- Bước đầu mỗi lượt có compile (~20 s khi cache ấm, ~150–200 s lần đầu); vài bước đầu chậm hơn vì Triton autotune lại.
