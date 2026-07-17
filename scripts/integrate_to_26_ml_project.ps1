param(
    [string]$Source = "D:\MI_project",
    [string]$Dest = "D:\26_ML_MIProject"
)

$ErrorActionPreference = "Stop"

function Assert-UnderPath {
    param(
        [string]$Path,
        [string]$Root
    )
    $resolvedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $resolvedPath.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify path outside destination: $resolvedPath"
    }
}

function Invoke-RoboCopyChecked {
    param(
        [string]$From,
        [string]$To,
        [string[]]$ExtraArgs = @()
    )
    if (-not (Test-Path -LiteralPath $From)) {
        Write-Host "skip missing $From"
        return
    }
    New-Item -ItemType Directory -Force -Path $To | Out-Null
    $args = @($From, $To, "/E", "/R:2", "/W:1", "/NFL", "/NDL", "/NP") + $ExtraArgs
    & robocopy @args | Out-Host
    $code = $LASTEXITCODE
    if ($code -ge 8) {
        throw "robocopy failed with exit code $code from $From to $To"
    }
}

$sourceFull = [System.IO.Path]::GetFullPath($Source).TrimEnd('\')
$destFull = [System.IO.Path]::GetFullPath($Dest).TrimEnd('\')

if (-not (Test-Path -LiteralPath $sourceFull)) {
    throw "Source does not exist: $sourceFull"
}
if (-not (Test-Path -LiteralPath $destFull)) {
    throw "Destination does not exist: $destFull"
}
if ($sourceFull.Equals($destFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Source and destination are the same directory."
}

Write-Host "Integrating from $sourceFull to $destFull"

# Copy result artifacts and supporting inputs. Exclude duplicated code copies and caches.
Invoke-RoboCopyChecked "$sourceFull\outputs" "$destFull\outputs" @("/XD", "$sourceFull\outputs\code", "$sourceFull\outputs\__pycache__", "/XF", "~$*")
Invoke-RoboCopyChecked "$sourceFull\inputs" "$destFull\inputs" @("/XD", "$sourceFull\inputs\__pycache__", "/XF", "~$*")
Invoke-RoboCopyChecked "$sourceFull\scripts" "$destFull\scripts" @("/XD", "$sourceFull\scripts\__pycache__", "/XF", "*.pyc")

# Keep a canonical copy of experimental runners under scripts only.
$canonicalScripts = @(
    "run_qwen3000_cpu_suite.py",
    "run_activation_patching_qwen.py"
)
foreach ($name in $canonicalScripts) {
    $from = Join-Path "$sourceFull\outputs\code" $name
    if (Test-Path -LiteralPath $from) {
        Copy-Item -LiteralPath $from -Destination (Join-Path "$destFull\scripts" $name) -Force
    }
}

if (Test-Path -LiteralPath "$sourceFull\package.json") {
    Copy-Item -LiteralPath "$sourceFull\package.json" -Destination "$destFull\package.json" -Force
}

# Remove duplicate experiment-running code locations in the destination.
$removed = New-Object System.Collections.Generic.List[string]
$duplicatePaths = @(
    "$destFull\outputs\code",
    "$destFull\run_qwen_no_plots.py"
)
foreach ($path in $duplicatePaths) {
    if (Test-Path -LiteralPath $path) {
        Assert-UnderPath -Path $path -Root $destFull
        Remove-Item -LiteralPath $path -Recurse -Force
        $removed.Add($path)
    }
}

if ($removed.Count -gt 0) {
    $removedText = $removed | ForEach-Object { "- `$_" } | Out-String
} else {
    $removedText = "- none found in destination; source `outputs/code/` was skipped during copy."
}

$manifest = @"
# Integration From D:\MI_project

Integrated on: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

Source: $sourceFull
Destination: $destFull

Copied:
- `outputs/` excluding duplicate `outputs/code/`
- `inputs/`
- `scripts/`
- `package.json`
- canonical runner scripts from source `outputs/code/` into destination `scripts/`

Canonical experiment-running code location:
- `scripts/run_qwen3000_cpu_suite.py`
- `scripts/run_activation_patching_qwen.py`
- `scripts/remote_seeta_job.py`
- `scripts/build_mmlu_raw_full_csv.py`
- `scripts/summarize_full_results.py`

Duplicate code removed from destination:
$removedText

Notes:
- Existing root-level project notebooks and the original `colab_sycophancy_locate_steer_improve.py` were kept.
- Generated outputs, reports, PPTX files, and QA renders are under `outputs/`.
"@

Set-Content -LiteralPath "$destFull\INTEGRATION_FROM_MI_PROJECT.md" -Value $manifest -Encoding UTF8

Write-Host "Integration complete."
Write-Host "Manifest: $destFull\INTEGRATION_FROM_MI_PROJECT.md"
