# Run tools inside the project environment without activating a global environment.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$mambaExe = Join-Path $projectRoot '.tools/Library/bin/micromamba.exe'
$environmentPrefix = Join-Path $projectRoot '.venv'
if (!(Test-Path -LiteralPath $mambaExe) -or !(Test-Path -LiteralPath "$environmentPrefix/python.exe")) {
    throw 'Environment is missing. Run scripts/bootstrap.ps1 first.'
}
if ($args.Count -eq 0) {
    throw 'Supply a command, for example: scripts/run.ps1 python -m ecologyhydro doctor'
}
$previousMambaRoot = $env:MAMBA_ROOT_PREFIX
$previousUvCache = $env:UV_CACHE_DIR
$previousPackageCache = $env:CONDA_PKGS_DIRS
try {
    $env:MAMBA_ROOT_PREFIX = Join-Path $projectRoot '.mamba'
    $env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
    $env:CONDA_PKGS_DIRS = Join-Path $projectRoot '.mamba/pkgs'
    Push-Location $projectRoot
    try {
        & $mambaExe run --prefix $environmentPrefix @args
        $commandExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    $env:MAMBA_ROOT_PREFIX = $previousMambaRoot
    $env:UV_CACHE_DIR = $previousUvCache
    $env:CONDA_PKGS_DIRS = $previousPackageCache
}
exit $commandExitCode
