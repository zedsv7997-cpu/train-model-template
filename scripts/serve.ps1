# Chạy một model GGUF bằng llama-server trong giới hạn VRAM, in số đo bộ nhớ, giữ server chạy đến khi Ctrl+C.
# API OpenAI-compatible tại http://127.0.0.1:<Port>/v1 (hỗ trợ tools / tool_calls nhờ --jinja).
#
#   powershell -ExecutionPolicy Bypass -File scripts\serve.ps1 -Model models\qwen35-2b-x-vi-en-IQ4_XS.gguf
#   powershell -ExecutionPolicy Bypass -File scripts\serve.ps1 -Model models\qwen35-2b-x-vi-en-IQ4_XS.gguf -Ngl 99
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [int]$Ngl = -1,                 # -1 = tự dò; 99 = mọi layer trên GPU
    [int]$VramLimitMiB = 1024,
    [int]$RamLimitMiB = 1024,
    [int]$Ctx = 12288,              # system prompt + vài chục tool đã ~6.4K token
    [int]$Batch = 256,
    [int]$UBatch = 128,
    [int]$Port = 8090,
    [ValidateSet("cuda-13", "cuda-12", "cpu")][string]$Backend = "cuda-13",
    [ValidateSet("q8_0", "q4_0", "f16")][string]$CacheType = "q8_0",
    [int]$CtxCheckpoints = 4,
    [switch]$EmbdOnGpu,
    [switch]$Mmap
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "llama_common.ps1")

$Model = (Resolve-Path $Model).Path
$opts = @{ Ctx = $Ctx; Batch = $Batch; UBatch = $UBatch; Port = $Port; Backend = $Backend; CacheType = $CacheType;
          CtxCheckpoints = $CtxCheckpoints; Mmap = $Mmap; EmbdOnGpu = $EmbdOnGpu }
if ($Ngl -lt 0) { $Ngl = Find-BestNgl -Model $Model -VramLimitMiB $VramLimitMiB -Opts $opts }

$base = Get-GpuUsedMiB
$server = Start-LlamaServer -Model $Model -Ngl $Ngl @opts
try {
    Invoke-WarmUp -Port $Port
    $m = Measure-LlamaServer -Server $server -BaselineMiB $base
    Write-Host ""
    Write-Host "Model : $Model ($Backend, -ngl $Ngl)"
    Write-Host ("VRAM  : {0} MiB (giới hạn {1})" -f $m.VramMiB, $VramLimitMiB)
    Write-Host ("RAM   : {0} MiB working set, {1} MiB trong Task Manager (giới hạn {2})" -f $m.RamMiB, $m.PrivWsMiB, $RamLimitMiB)
    Show-BufferSizes $server
    if ($m.VramMiB -gt $VramLimitMiB -or $m.RamMiB -gt $RamLimitMiB) { Write-Warning "Vượt giới hạn bộ nhớ!" }
    Write-Host ""
    Write-Host "Server: http://127.0.0.1:$Port/v1  (log: $($server.Log)). Ctrl+C để dừng."
    Wait-Process -Id $server.Process.Id
} finally {
    Stop-LlamaServer $server
}
