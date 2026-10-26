# Tải llama.cpp bản dựng sẵn cho Windows (dùng để chạy model, đo bộ nhớ và đánh giá).
# Driver hiện tại hỗ trợ CUDA 13.x nên mặc định lấy bản cuda-13.
# Chạy:  powershell -ExecutionPolicy Bypass -File setup\get_llamacpp.ps1
param(
    [ValidateSet("cuda-13", "cuda-12", "cpu")]  # không lấy bản vulkan: không chạy gì trên iGPU Radeon
    [string]$Backend = "cuda-13",
    [string]$Dest
)
if (-not $Dest) { $Dest = Join-Path $PSScriptRoot "..\tools\llama.cpp\$Backend" }
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest chậm hẳn khi hiện thanh tiến trình
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Bản "latest" có thể không chứa file build, nên duyệt các release gần nhất để tìm bản có file Windows
$releases = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20" `
    -Headers @{ "User-Agent" = "train-model-template" }
$asset = $null
foreach ($r in $releases) {
    $asset = $r.assets | Where-Object { $_.name -like "llama-*-bin-win-$Backend*-x64.zip" } | Select-Object -First 1
    if ($asset) { $release = $r; break }
}
if (-not $asset) { throw "Không tìm thấy bản Windows '$Backend' trong 20 release gần nhất." }

$downloads = @($asset)
if ($asset.name -match "cuda-([\d\.]+)-x64") {
    $cudart = $release.assets | Where-Object { $_.name -eq "cudart-llama-bin-win-cuda-$($Matches[1])-x64.zip" }
    if (-not $cudart) { throw "Không tìm thấy gói cudart cho CUDA $($Matches[1])." }
    $downloads += $cudart
}

New-Item -ItemType Directory -Force $Dest | Out-Null
foreach ($a in $downloads) {
    $zip = Join-Path $env:TEMP $a.name
    Write-Host ("Tải {0} ({1:N0} MB) từ release {2}" -f $a.name, ($a.size / 1MB), $release.tag_name)
    Invoke-WebRequest $a.browser_download_url -OutFile $zip
    Expand-Archive $zip -DestinationPath $Dest -Force
    Remove-Item $zip
}
Set-Content (Join-Path $Dest "release.txt") "$($release.tag_name) $($asset.name)" -Encoding utf8

$server = Get-ChildItem $Dest -Recurse -Filter llama-server.exe | Select-Object -First 1
if (-not $server) { throw "Giải nén xong nhưng không thấy llama-server.exe" }
Write-Host "OK: $($server.FullName)"
& $server.FullName --version
