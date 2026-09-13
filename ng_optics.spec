# Build from NG_v2 with: python -m PyInstaller --noconfirm ng_optics.spec
from pathlib import Path

root = Path(SPECPATH)
a = Analysis(
    [str(root / 'main.py')],
    pathex=[str(root)],
    binaries=[],
    datas=[(str(root / 'locales'), 'locales')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'PySide6', 'PyQt5', 'tkinter'],
    noarchive=False,
)
# Qt 6.11 uses Windows' ICU API. A similarly named ICU from tools on PATH
# (for example Poppler) has different exports and breaks QtCore at startup.
# Let Windows resolve its own ICU instead of packaging those foreign DLLs.
a.binaries = [entry for entry in a.binaries
              if not Path(entry[0]).name.lower().startswith('icu')]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='NG Optics v0.2.0',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    version=str(root / 'windows_version.txt'),
)
