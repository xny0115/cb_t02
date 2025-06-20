import importlib
import subprocess
import sys
import re
from pathlib import Path

req_file = Path('requirements.txt')
if not req_file.exists():
    sys.exit(0)

pkgs = [line.strip().lstrip('\ufeff') for line in req_file.read_text().splitlines() if line.strip() and not line.startswith('#')]

for pkg in pkgs:
    name = re.split(r'[<>=]', pkg)[0]
    try:
        importlib.import_module(name)
        continue
    except Exception:
        pass
    install = pkg.split('+')[0]
    subprocess.run([sys.executable, '-m', 'pip', 'install', install], check=False)
