$ErrorActionPreference = "Stop"

$train = "data/raw/train.csv"
$later = "data/raw/later.csv"
$holdout = "data/raw/holdout_features.csv"

$expected = @{
    "train.csv" = "9aea26c5aa76831f4a4e58a3dfafd06d47144136c13ace14834010bfcf472781"
    "later.csv" = "4c161defd6502ae4c6336aef612118ac329579730f31ee879d265e5df692db8c"
    "holdout_features.csv" = "c701755d2e84f1e877edf12a2aab797bba448807816635c8d4ceded8226cf896"
}

foreach ($path in @($train, $later, $holdout)) {
    if (-not (Test-Path $path)) {
        throw "Missing $path. Copy the original CSV from the faculty ZIP without opening/saving it in Excel."
    }
    $name = [System.IO.Path]::GetFileName($path)
    $actual = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
    if ($actual -ne $expected[$name]) {
        throw "SHA-256 mismatch for $path. Expected $($expected[$name]), got $actual. Restore the original file from the faculty ZIP."
    }
    Write-Host "$path hash verified: $actual"
}

if (-not (Test-Path ".dvc")) {
    dvc init
}

dvc add $train $later

Write-Host ""
Write-Host "DVC tracking created. Keep the CSVs out of Git; commit the .dvc pointer files and DVC config."
Write-Host "Per the handbook layout, keep data/raw/holdout_features.csv in the repository."
Write-Host "Next inspect: git status"
