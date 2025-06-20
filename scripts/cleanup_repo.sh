#!/usr/bin/env bash
git rm -r --cached models node_modules dist *.log || true
git add .gitignore
echo "[cleanup] done"
