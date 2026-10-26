# Bản merge HF -> GGUF BF16 -> imatrix (từ chính hội thoại của dữ liệu, data\calib.txt) -> các bản nén. Chạy trên Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts\quantize.ps1 -Merged outputs\qwen35-2b-<Tag>\merged-vi-en -Out models\qwen35-2b-<Tag>-vi-en
#   ... -Quants IQ4_XS,Q4_K_M,Q5_K_S
#   ... -EmbType q5_K     # ép bảng embedding (2B: ~508M tham số, nằm trên RAM) xuống q5_K để bớt ~65MB RAM
param(
    [Parameter(Mandatory = $true)][string]$Merged,
    [Parameter(Mandatory = $true)][string]$Out,
    [string[]]$Quants = @("IQ4_XS"),  # Q4_K_M không đúng hơn IQ4_XS mà tốn thêm ~90 MiB RAM
    [string]$Calib = "data\calib.txt",  # văn bản hội thoại đã render (vài trăm mẫu) cho llama-imatrix
    [string]$EmbType = ""
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "llama_common.ps1")
$bin = Join-Path $ProjectRoot "tools\llama.cpp\cuda-13"
$convert = Join-Path $ProjectRoot "tools\llama.cpp-src\convert_hf_to_gguf.py"
if (-not (Test-Path $convert)) { throw "Thiếu $convert. Chạy: powershell -ExecutionPolicy Bypass -File setup\get_llamacpp_src.ps1" }
$Quants = @($Quants | ForEach-Object { $_ -split "," } | Where-Object { $_ })

New-Item -ItemType Directory -Force (Split-Path $Out -Parent) | Out-Null
$bf16 = "$Out-BF16.gguf"
$imatrix = "$Out-imatrix.gguf"

# Ghi ra file tạm, xong (mã thoát 0) mới đổi tên: bị dừng giữa chừng thì không để lại BF16 cụt hay imatrix dở
# (llama-imatrix ghi đè file -o sau mỗi 10 chunk) mà lần chạy sau tưởng là xong và dùng lại; GGUF nén dở thì
# app nào tự lấy file *-IQ4_XS.gguf mới nhất có thể chép nhầm.
function Complete-Part([string]$Dst) { Move-Item -Force ($Dst -replace '\.gguf$', '.part.gguf') $Dst }

if (-not (Test-Path $bf16)) {
    Write-Host "==> Convert sang GGUF BF16 (chỉ phần ngôn ngữ, bỏ vision)"
    # qua wrapper để convert được cả model đã tỉa từ vựng (prune_vocab.py)
    & $VenvPython (Join-Path $PSScriptRoot "convert_gguf.py") $Merged --outtype bf16 --outfile ($bf16 -replace '\.gguf$', '.part.gguf')
    if ($LASTEXITCODE -ne 0) { throw "convert_hf_to_gguf.py lỗi" }
    Complete-Part $bf16
}

if (-not (Test-Path $imatrix)) {
    # -c 8192: mỗi chunk chứa trọn một hội thoại (system prompt + tool + các lượt), giống hệt lúc chạy thật
    Write-Host "==> Tính imatrix từ $Calib"
    & (Join-Path $bin "llama-imatrix.exe") -m $bf16 -f $Calib -o ($imatrix -replace '\.gguf$', '.part.gguf') -c 8192 -b 2048 -ngl 99
    if ($LASTEXITCODE -ne 0) { throw "llama-imatrix lỗi" }
    Complete-Part $imatrix
}

$suffix = ""; $extra = @()
if ($EmbType) { $suffix = "-emb$EmbType"; $extra = @("--token-embedding-type", $EmbType) }
foreach ($q in $Quants) {
    $dst = "$Out-$q$suffix.gguf"
    Write-Host "==> Nén $q -> $dst"
    & (Join-Path $bin "llama-quantize.exe") --imatrix $imatrix @extra $bf16 ($dst -replace '\.gguf$', '.part.gguf') $q
    if ($LASTEXITCODE -ne 0) { throw "llama-quantize lỗi ($q)" }
    Complete-Part $dst
}
Get-ChildItem "$Out-*.gguf" | Where-Object { $_.Name -notlike "*.part.gguf" } | Select-Object Name, @{ n = "MiB"; e = { [int]($_.Length / 1MB) } } | Format-Table -AutoSize
