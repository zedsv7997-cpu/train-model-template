# Hàm dùng chung cho các script: đường dẫn Python, chạy llama-server nền, đo VRAM/RAM, tự dò -ngl.
# VRAM = memory.used của GPU sau khi chạy server trừ đi lúc trước khi chạy (GPU rời của laptop thường rảnh = 0).
# RAM  = working set của tiến trình llama-server (gồm cả DLL; con số Task Manager hiện là PrivWS, nhỏ hơn).
#
# Qwen3.5 dùng chung bảng embedding cho đầu vào và lớp output (~508M tham số với bản 2B). Nếu để llama.cpp
# đưa lớp output lên GPU, bảng này bị nhân đôi (một bản GPU + một bản RAM). Vì vậy mặc định ép nó ở CPU
# (-ot token_embd\.weight=CPU): mọi layer lên GPU mà không tốn thêm ~400MB VRAM.

$ProjectRoot = Split-Path $PSScriptRoot -Parent
# venv train (torch, Unsloth...). Đường dẫn tuyệt đối: tiến trình chạy nền qua WMI có thể thiếu PATH của user
$VenvPython = Join-Path $env:USERPROFILE "ml-env\Scripts\python.exe"

function Find-LlamaServer {
    param([string]$Backend = "cuda-13")
    $exe = Get-ChildItem (Join-Path $ProjectRoot "tools\llama.cpp\$Backend") -Recurse -Filter llama-server.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $exe) { throw "Chưa có llama-server.exe ($Backend). Chạy: powershell -ExecutionPolicy Bypass -File setup\get_llamacpp.ps1 -Backend $Backend" }
    return $exe.FullName
}

function Get-PythonCmd {
    # script chấm điểm chỉ cần thư viện chuẩn: không có venv thì Python nào cũng được
    if (Test-Path $VenvPython) { return $VenvPython }
    if (Get-Command py -ErrorAction SilentlyContinue) { return "py" }
    if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
    throw "Không tìm thấy Python trên Windows."
}

function Get-GpuUsedMiB {
    $v = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1
    return [int]"$v".Trim()
}

function Start-LlamaServer {
    param(
        [Parameter(Mandatory = $true)][string]$Model,
        [int]$Ngl = 99, [int]$Ctx = 12288, [int]$Batch = 256, [int]$UBatch = 128, [int]$Port = 8090,
        [string]$Backend = "cuda-13", [string]$CacheType = "q8_0", [int]$CtxCheckpoints = 4, [switch]$Mmap, [switch]$EmbdOnGpu,
        [string]$Device = "", [int]$Threads = 0,  # vd. -Device none -Threads 6: chạy hẳn trên CPU
        [string[]]$ExtraArgs
    )
    $exe = Find-LlamaServer $Backend
    # llama-server mở cổng với SO_REUSEADDR: cổng đã có server khác (serve.ps1 cũng mặc định 8090, hay llama-server mồ côi
    # của lượt bị dừng) thì nó vẫn chạy, /health trả 200 từ server CŨ và request chia ngẫu nhiên -> chấm nhầm model
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Cổng $Port đang có tiến trình khác nghe (llama-server cũ?). Tắt nó hoặc đổi -Port."
    }
    $logDir = Join-Path $ProjectRoot "results\logs"
    New-Item -ItemType Directory -Force $logDir | Out-Null
    $where = if ($Device) { "dev-$Device" } else { "ngl$Ngl" }
    $log = Join-Path $logDir ("{0}-{1}-{2}.log" -f [IO.Path]::GetFileNameWithoutExtension($Model), $Backend, $where)
    # -np 1: một slot, tránh nhân bản KV cache / recurrent state
    # -b/-ub nhỏ: giảm compute buffer
    # -cram 0: tắt prompt cache trên RAM (mặc định tới 8GB)
    # -fit off: không để llama.cpp tự đổi tham số, giữ đúng -ngl/-c đã chọn
    $a = @("-m", "`"$Model`"", "-ngl", $Ngl, "-c", $Ctx, "-b", $Batch, "-ub", $UBatch,
           "-np", 1, "-cram", 0, "-fit", "off", "--jinja", "--host", "127.0.0.1", "--port", $Port)
    if ($Device) { $a += @("--device", $Device) }
    if ($Threads -gt 0) { $a += @("-t", $Threads) }
    if (-not $EmbdOnGpu) { $a += @("-ot", "token_embd\.weight=CPU") }
    # KV cache q8_0: giảm một nửa bộ nhớ context (prompt agent nhiều tool ~6.4K token nên cần -c 12288)
    if ($CacheType -and $CacheType -ne "f16") { $a += @("-ctk", $CacheType, "-ctv", $CacheType) }
    # Mặc định llama.cpp giữ tới 32 checkpoint/slot; với Qwen3.5 (lai DeltaNet) mỗi cái là một bản trạng thái ~20MB trên RAM,
    # nên RAM phình dần theo số lượt (đo: 969 -> 1099 MiB sau 556 request). Agent chỉ nối thêm tin nhắn nên vài cái là đủ.
    if ($CtxCheckpoints -ge 0) { $a += @("-ctxcp", $CtxCheckpoints) }
    # -lm none: nạp hẳn vào RAM để số đo trung thực; mmap làm working set dao động theo cache của Windows
    if ($Mmap) { $a += @("-lm", "mmap") } else { $a += @("-lm", "none") }
    # cờ thêm, vd. speculative decoding: --spec-type draft-mtp
    if ($ExtraArgs) { $a += $ExtraArgs }
    $proc = Start-Process -FilePath $exe -ArgumentList $a -PassThru -NoNewWindow `
        -RedirectStandardError $log -RedirectStandardOutput "$log.stdout"
    $server = [pscustomobject]@{ Process = $proc; Log = $log; Port = $Port; Ngl = $Ngl }
    $deadline = (Get-Date).AddSeconds(300)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) {
            throw ("llama-server thoát sớm (ngl={0}). Log: {1}`n{2}" -f $Ngl, $log, ((Get-Content $log -Tail 15) -join "`n"))
        }
        try {
            $r = Invoke-WebRequest "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { return $server }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    Stop-LlamaServer $server
    throw "llama-server không sẵn sàng sau 300 giây. Log: $log"
}

function Stop-LlamaServer {
    param($Server)
    if ($Server -and -not $Server.Process.HasExited) {
        Stop-Process -Id $Server.Process.Id -Force -ErrorAction SilentlyContinue
        $Server.Process.WaitForExit(15000) | Out-Null
    }
    Start-Sleep -Seconds 1   # chờ driver trả VRAM
}

function Invoke-WarmUp {
    # Một prompt đủ dài để cuBLAS / compute buffer được cấp phát hết trước khi đo
    param([int]$Port)
    $body = @{
        messages    = @(@{ role = "user"; content = ("Tell me about the weather today. " * 40) })
        max_tokens  = 8
        temperature = 0
    } | ConvertTo-Json -Depth 5
    Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -Body $body `
        -ContentType "application/json" -TimeoutSec 600 | Out-Null
}

function Measure-LlamaServer {
    param($Server, [int]$BaselineMiB)
    $p = Get-Process -Id $Server.Process.Id
    $privWs = (Get-CimInstance Win32_PerfFormattedData_PerfProc_Process -Filter "IDProcess=$($p.Id)").WorkingSetPrivate
    return [pscustomobject]@{
        VramMiB    = (Get-GpuUsedMiB) - $BaselineMiB
        RamMiB     = [int]($p.WorkingSet64 / 1MB)
        RamPeakMiB = [int]($p.PeakWorkingSet64 / 1MB)
        PrivWsMiB  = [int]($privWs / 1MB)
    }
}

function Show-BufferSizes {
    # Dòng "... buffer size = ..." trong log cho biết từng phần nằm ở GPU hay CPU
    param($Server)
    Select-String -Path $Server.Log -Pattern "buffer size|offloaded" |
        ForEach-Object { Write-Host ("    " + ($_.Line.Trim() -replace '^\S+\s+\S+\s+', '')) }
}

function Test-Ngl {
    param([string]$Model, [int]$Ngl, [hashtable]$Opts)
    $base = Get-GpuUsedMiB
    $s = Start-LlamaServer -Model $Model -Ngl $Ngl @Opts
    try {
        Invoke-WarmUp -Port $s.Port
        $m = Measure-LlamaServer -Server $s -BaselineMiB $base
        $nLayer = $null
        $hit = Select-String -Path $s.Log -Pattern "offloaded \d+/(\d+) layers" | Select-Object -First 1
        if ($hit) { $nLayer = [int]$hit.Matches[0].Groups[1].Value }
        Write-Host ("  ngl={0,-3} VRAM={1,5} MiB  RAM={2,5} MiB (TaskMgr {3})" -f $Ngl, $m.VramMiB, $m.RamMiB, $m.PrivWsMiB)
        return [pscustomobject]@{ Ngl = $Ngl; VramMiB = $m.VramMiB; RamMiB = $m.RamMiB; NLayer = $nLayer }
    } finally {
        Stop-LlamaServer $s
    }
}

function Find-BestNgl {
    # Thử đưa hết lên GPU trước; không vừa thì tìm nhị phân -ngl lớn nhất mà VRAM vẫn <= giới hạn
    # (VRAM tăng dần theo -ngl)
    # Chừa $MarginMiB: VRAM đo ngay sau khởi động còn tăng vài MiB khi chạy lâu (Q4_K_M -ngl 25: 1024 -> 1030 MiB)
    param([string]$Model, [int]$VramLimitMiB, [hashtable]$Opts = @{}, [int]$MarginMiB = 24)
    $VramLimitMiB -= $MarginMiB
    Write-Host "Dò -ngl cho $([IO.Path]::GetFileName($Model)) (VRAM <= $VramLimitMiB MiB, đã chừa $MarginMiB MiB):"
    $full = Test-Ngl -Model $Model -Ngl 99 -Opts $Opts
    if ($full.VramMiB -le $VramLimitMiB) { Write-Host "  => chọn -ngl 99 (mọi layer trên GPU)"; return 99 }
    $lo = 0; $best = 0
    $hi = 32
    if ($full.NLayer) { $hi = $full.NLayer - 1 }
    while ($lo -le $hi) {
        $mid = [int][math]::Floor(($lo + $hi) / 2)
        $t = Test-Ngl -Model $Model -Ngl $mid -Opts $Opts
        if ($t.VramMiB -le $VramLimitMiB) { $best = $mid; $lo = $mid + 1 } else { $hi = $mid - 1 }
    }
    Write-Host "  => chọn -ngl $best"
    return $best
}
