"""Test a copied EXE with no Python directories on PATH."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parents[1]
exe = root / 'dist' / 'NG Optics v0.2.0.exe'
with tempfile.TemporaryDirectory(prefix='ng-optics-check-') as directory:
    folder = Path(directory)
    shutil.copy2(exe, folder / exe.name)
    report = folder / 'report.json'
    environment = {key: value for key, value in os.environ.items()
                   if key.upper() not in ('PYTHONPATH', 'PYTHONHOME', 'QT_PLUGIN_PATH',
                                          'QML2_IMPORT_PATH', 'PATH')}
    environment['PATH'] = str(Path(os.environ['SYSTEMROOT']) / 'System32')
    environment['QT_QPA_PLATFORM'] = 'offscreen'
    # Capture failures immediately: a windowed bootloader can otherwise leave
    # an error dialog open while the automated test waits for it to exit.
    error_path = folder / 'stderr.log'
    with error_path.open('wb') as error_log:
        process = subprocess.Popen([str(folder / exe.name), '--smoke-test', str(report)],
                                   cwd=folder, env=environment, stderr=error_log,
                                   stdout=subprocess.DEVNULL)
        deadline = time.monotonic() + 60
        while process.poll() is None:
            if error_path.stat().st_size or time.monotonic() > deadline:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               capture_output=True)
                process.wait(timeout=10)
                raise RuntimeError(error_path.read_text(errors='replace') or 'Smoke test timed out')
            time.sleep(0.1)
    if process.returncode:
        raise RuntimeError(error_path.read_text(errors='replace'))
    print(report.read_text(encoding='utf-8'))
    shutil.copy2(report, exe.parent / 'smoke-report.json')
print(f'EXE size: {exe.stat().st_size / 1024**2:.1f} MiB')
