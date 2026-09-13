$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python -m PyInstaller --noconfirm ng_optics.spec
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed' }
    Write-Host 'Built dist\NG Optics v0.2.0.exe'
} finally {
    Pop-Location
}
