#!/usr/bin/env bash
set -euo pipefail
FLAG=/tmp/deps_ready_cb_t02
if [[ -f $FLAG ]]; then echo "[bootstrap] deps ready"; exit 0; fi

# ----- Python -----
python - <<'PY'
import subprocess, sys, importlib.util as iu
pkgs = ["torch==2.3.*","torchtext==0.18.*","pandas>=2.1","psutil>=5.9","pytest","pytest-cov"]
for spec in pkgs:
    mod = spec.split("==")[0].split(">=")[0]
    if iu.find_spec(mod) is None:
        subprocess.check_call([sys.executable,"-m","pip","install",spec])
PY

# ----- Node -----
if command -v npm >/dev/null; then
    npm install --silent jest@29 babel-jest@29 @babel/core@7 @babel/preset-env@7 \
        @babel/preset-react@7 react@18 react-dom@18 jsdom@22
fi

touch $FLAG
echo "[bootstrap] completed"
