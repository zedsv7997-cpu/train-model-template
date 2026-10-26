# Chạy train.py, tự chạy lại (tiếp từ checkpoint) nếu tiến trình chết giữa chừng — Python trên Windows thỉnh thoảng
# crash "access violation" khi dọn rác, thoát mã 5 / -1073741819 mà không có traceback.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_train.ps1 -Out outputs\qwen35-2b-x -Log outputs\train-x `
#       -TrainArgs "--train data\train.jsonl --no-samples"
param(
    # Công thức train (dữ liệu, độ dài chuỗi, 16 hội thoại/bước, LR, MXFP8...) là mặc định của train.py
    [string]$TrainArgs = "--no-samples",
    [string]$Out = "outputs\qwen35-2b-lora",
    [string]$Log = "outputs\train-2b",
    [int]$MaxAttempts = 6,
    # Tiến trình khác giữ VRAM (vd. llama-server của app ~0.85 GB) làm tổng vượt 8 GB -> WDDM âm thầm đẩy sang RAM, train
    # chậm ~8 lần (đo: 58-62 s/bước thay vì 7 s) -> chờ GPU rảnh rồi mới train. Từ khi cài lại Windows (2026-10-26)
    # màn hình chạy trên RTX (Disp.A On): desktop giữ sẵn ~1.1 GB (lúc rảnh trước đây 0 MiB) -> ngưỡng 1300 thay vì 300;
    # vẫn bắt được llama-server của app (desktop + server ~1.9 GB)
    [int]$GpuFreeMiB = 1300,
    [int]$GpuWaitMin = 120          # chờ tối đa; quá thì vẫn train (train.py tự dừng nếu bước chậm bất thường)
)
. (Join-Path $PSScriptRoot "llama_common.ps1")
Set-Location $ProjectRoot
$env:HF_HUB_OFFLINE = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"; $env:PYTHONFAULTHANDLER = "1"

function Write-RunnerLog([string]$Msg) {
    "$(Get-Date -Format s) $Msg" | Add-Content "$Log.runner.log" -Encoding utf8
}

function Wait-GpuFree {
    $deadline = (Get-Date).AddMinutes($GpuWaitMin)
    $warned = $false
    while ($true) {
        $used = Get-GpuUsedMiB
        if ($used -le $GpuFreeMiB) {
            if ($warned) { Write-RunnerLog "GPU đã rảnh ($used MiB)" }
            return
        }
        if ((Get-Date) -gt $deadline) {
            Write-RunnerLog "Chờ quá $GpuWaitMin phút, GPU vẫn bị chiếm $used MiB -> train luôn"
            return
        }
        if (-not $warned) {
            $apps = (& nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader) -join "; "
            Write-RunnerLog "GPU đang bị chiếm $used MiB ($apps) -> chờ giải phóng (tắt llama-server của app)"
            $warned = $true
        }
        Start-Sleep -Seconds 30
    }
}

# Gọi lại để merge lại (khi bản merge hỏng): train.py chỉ merge, out.log mới sẽ đè log train
# -> giữ log của lần train một lần
if ((Test-Path (Join-Path $Out "lora\TRAIN_DONE")) -and (Test-Path "$Log.out.log") -and -not (Test-Path "$Log.out.train.log")) {
    Copy-Item "$Log.out.log" "$Log.out.train.log"
    Copy-Item "$Log.err.log" "$Log.err.train.log" -ErrorAction SilentlyContinue
}
for ($i = 1; $i -le $MaxAttempts; $i++) {
    Wait-GpuFree
    Write-RunnerLog "Lần chạy $i"
    $p = Start-Process -FilePath $VenvPython -ArgumentList "scripts\train.py --out $Out $TrainArgs" -WorkingDirectory $ProjectRoot `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$Log.out.log" -RedirectStandardError "$Log.err.log"
    if (Test-Path (Join-Path $Out "merged\config.json")) {
        Write-RunnerLog "Xong (mã thoát $($p.ExitCode))"
        exit 0
    }
    $why = if ($p.ExitCode -eq 3) { "dừng vì bước train chậm bất thường (VRAM tràn?)" } else { "tiến trình chết (mã $($p.ExitCode))" }
    Write-RunnerLog "$why, chạy lại từ checkpoint"
    # lần sau ghi đè -> giữ cả hai (out.log có loss và lý do SlowGuard dừng)
    Copy-Item "$Log.err.log" "$Log.err.attempt$i.log" -Force -ErrorAction SilentlyContinue
    Copy-Item "$Log.out.log" "$Log.out.attempt$i.log" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 10
}
Write-RunnerLog "Bỏ cuộc sau $MaxAttempts lần"
exit 1
