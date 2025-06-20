# Installation Guide

## Requirements
- Python 3.10
- CUDA 11.8 compatible GPU
- Windows 10 or later

## Setup Steps
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure a Java JDK is installed and `JAVA_HOME` is configured.
   Komoran relies on Java to run. Verify with `javac -version`.
3. Set environment variables as needed:
   - `PYTHONIOENCODING=utf-8`
   - `SOYNLP_DATA_DIR` (optional path for soynlp data cache)

Run `python run.py` to start the application.
