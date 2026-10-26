# Tải mã nguồn llama.cpp CÙNG phiên bản với file .exe đã tải, để có convert_hf_to_gguf.py (+ gguf-py đi kèm).
# convert_hf_to_gguf.py tự nạp gguf-py nằm cạnh nó, nên không cần pip install gì thêm.
# Chạy:  powershell -ExecutionPolicy Bypass -File setup\get_llamacpp_src.ps1
param(
    [string]$Dest = (Join-Path $PSScriptRoot "..\tools\llama.cpp-src"),
    [string]$Tag = ""  # vd. b11803; bỏ trống = đọc từ tools\llama.cpp\cuda-13\release.txt
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$relFile = Join-Path $PSScriptRoot "..\tools\llama.cpp\cuda-13\release.txt"
$tag = if ($Tag) { $Tag } else { "master" }
if (-not $Tag -and (Test-Path $relFile)) { $tag = ((Get-Content $relFile -Raw).Trim() -split " ")[0] }
if ($tag -eq "master") { $url = "https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.zip" }
else { $url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$tag.zip" }

$zip = Join-Path $env:TEMP "llama.cpp-src-$tag.zip"
$tmp = Join-Path $env:TEMP "llama.cpp-src-$tag"
Write-Host "Tải mã nguồn llama.cpp $tag"
Invoke-WebRequest $url -OutFile $zip
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
Expand-Archive $zip -DestinationPath $tmp -Force
if (Test-Path $Dest) { Remove-Item $Dest -Recurse -Force }
Move-Item (Get-ChildItem $tmp -Directory | Select-Object -First 1).FullName $Dest
Remove-Item $zip, $tmp -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "OK: $(Join-Path $Dest 'convert_hf_to_gguf.py')"
