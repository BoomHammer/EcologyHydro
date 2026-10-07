# Reproduce the tested win-64 native stack; no changes to global Python/conda.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$mambaExe = Join-Path $projectRoot '.tools/Library/bin/micromamba.exe'
$environmentPrefix = Join-Path $projectRoot '.venv'
$lockPath = Join-Path $projectRoot 'locks/conda-win-64.lock'
$archive = Join-Path $projectRoot '.tmp/micromamba-2.9.0-0.tar.bz2'
$expectedHash = '97A336F4AB794BD96A6A4DA5E6ED63E75A1D31830414A182419B23D3B36F3FE0'
if (![Environment]::Is64BitOperatingSystem -or $env:OS -ne 'Windows_NT') {
    throw 'This lock file supports Windows x86-64 only.'
}
if (!(Test-Path -LiteralPath $lockPath)) {
    throw "Missing environment lock: $lockPath"
}
if (!(Test-Path -LiteralPath $mambaExe)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $archive), "$projectRoot/.tools" | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri 'https://conda.anaconda.org/conda-forge/win-64/micromamba-2.9.0-0.tar.bz2' -OutFile $archive
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedHash) {
        throw 'micromamba archive SHA256 mismatch; extraction cancelled.'
    }
    & tar -xjf $archive -C "$projectRoot/.tools" Library/bin/micromamba.exe
    if ($LASTEXITCODE -ne 0) { throw 'micromamba extraction failed.' }
}
$previousMambaRoot = $env:MAMBA_ROOT_PREFIX
$previousUvCache = $env:UV_CACHE_DIR
$previousPackageCache = $env:CONDA_PKGS_DIRS
try {
    $env:MAMBA_ROOT_PREFIX = Join-Path $projectRoot '.mamba'
    $env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
    $env:CONDA_PKGS_DIRS = Join-Path $projectRoot '.mamba/pkgs'
    if (!(Test-Path -LiteralPath "$environmentPrefix/conda-meta/history")) {
        if (Test-Path -LiteralPath $environmentPrefix) {
            throw 'An existing non-conda .venv was found; choose a new project checkout.'
        }
        & $mambaExe create --yes --prefix $environmentPrefix --file $lockPath
    } else {
        $installedLines = & $mambaExe env export --prefix $environmentPrefix --explicit
        if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the existing environment.' }
        $expectedPackages = @(Get-Content -LiteralPath $lockPath | Where-Object { $_ -match '^https://' })
        $installedPackages = @($installedLines | Where-Object { $_ -match '^https://' })
        if (Compare-Object $expectedPackages $installedPackages) {
            throw 'Existing environment differs from the lock. Recreate in a fresh checkout or review the dependency changes.'
        }
        Write-Output 'Existing native environment matches the lock.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Locked environment installation failed.' }
    & $mambaExe run --prefix $environmentPrefix uv pip install --python "$environmentPrefix/python.exe" --no-deps --no-build-isolation --offline --editable $projectRoot
    if ($LASTEXITCODE -ne 0) { throw 'Editable project installation failed.' }
    & $mambaExe run --prefix $environmentPrefix uv pip check --python "$environmentPrefix/python.exe"
    if ($LASTEXITCODE -ne 0) { throw 'Environment dependency check failed.' }
} finally {
    $env:MAMBA_ROOT_PREFIX = $previousMambaRoot
    $env:UV_CACHE_DIR = $previousUvCache
    $env:CONDA_PKGS_DIRS = $previousPackageCache
}
Write-Output 'Environment ready. Run scripts/run.ps1 python -m ecologyhydro doctor'
