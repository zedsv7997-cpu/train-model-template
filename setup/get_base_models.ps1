# Tải bản GGUF gốc (chưa fine-tune) của Unsloth làm mốc so sánh:
#   - Qwen3.5-2B  IQ4_XS (~1.17 GB): kiểm tra trước là bản 2B nén có vừa 1GB VRAM + 1GB RAM không
#   - Qwen3.5-0.8B Q8_0            : đối thủ cần vượt qua
# Chạy:  powershell -ExecutionPolicy Bypass -File setup\get_base_models.ps1
param(
    [string[]]$Files = @(
        "unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-IQ4_XS.gguf",
        "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
    ),
    [string]$Dest = (Join-Path $PSScriptRoot "..\models\base")
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $Dest | Out-Null

foreach ($f in $Files) {
    $parts = $f -split "/"
    $repo = "$($parts[0])/$($parts[1])"
    $file = $parts[2]
    $out = Join-Path $Dest $file
    if (Test-Path $out) { Write-Host "Đã có: $out"; continue }
    $url = "https://huggingface.co/$repo/resolve/main/$file"
    Write-Host "Tải $url"
    & curl.exe -L --fail --retry 3 -o "$out.part" $url
    if ($LASTEXITCODE -ne 0) { Remove-Item "$out.part" -ErrorAction SilentlyContinue; throw "Tải lỗi: $url" }
    Move-Item "$out.part" $out
}
Get-ChildItem $Dest -Filter *.gguf | Select-Object Name, @{ n = "MiB"; e = { [int]($_.Length / 1MB) } } | Format-Table -AutoSize
