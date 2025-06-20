#!/usr/bin/env bash
set -e
git rm -r --cached node_modules dist models checkpoints || true
git add .gitignore
echo "[cleanup] repository sanitized"
