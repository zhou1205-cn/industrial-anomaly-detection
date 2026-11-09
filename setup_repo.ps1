# ---------------------------------------------------------------
#  Set up repository layout from existing PyCharmMiscProject files
#  Copies (does NOT move) your .py / .csv files into src/ and results/
#  Your original files stay untouched.
# ---------------------------------------------------------------

$src = "C:\Users\ZPX\PyCharmMiscProject"
$dst = $PSScriptRoot

Write-Host ""
Write-Host "Source      : $src"
Write-Host "Repo root   : $dst"
Write-Host ""

if (-not (Test-Path $src)) {
    Write-Host "[ERROR] Source folder not found: $src" -ForegroundColor Red
    Write-Host "        Edit the `$src variable at the top of this script."
    exit 1
}

# --- create folders ---
foreach ($d in @("src", "results", "docs\figures")) {
    $p = Join-Path $dst $d
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
}

# --- copy python scripts ---
$pyFiles = Get-ChildItem -Path $src -Filter *.py -File
$copiedPy = 0
foreach ($f in $pyFiles) {
    $target = Join-Path $dst "src\$($f.Name)"
    if (-not (Test-Path $target)) {
        Copy-Item $f.FullName $target
        Write-Host "  + src\$($f.Name)"
        $copiedPy++
    } else {
        Write-Host "  = src\$($f.Name)  (already exists, skipped)" -ForegroundColor DarkGray
    }
}

# --- copy result csv ---
$csvFiles = Get-ChildItem -Path $src -Filter *.csv -File
$copiedCsv = 0
foreach ($f in $csvFiles) {
    $target = Join-Path $dst "results\$($f.Name)"
    if (-not (Test-Path $target)) {
        Copy-Item $f.FullName $target
        Write-Host "  + results\$($f.Name)"
        $copiedCsv++
    } else {
        Write-Host "  = results\$($f.Name)  (already exists, skipped)" -ForegroundColor DarkGray
    }
}

# --- report html: copy it manually into docs\ if you want it in the repo ---

Write-Host ""
Write-Host "Done. $copiedPy python file(s), $copiedCsv csv file(s) copied."
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Open src\*.py and update the DATA path if needed, e.g."
Write-Host "       DATA = Path(r`"D:\datasets\mvtec_ad`")"
Write-Host "  2. Then:  git init  ->  git add .  ->  git commit -m `"Initial commit`""
Write-Host ""
