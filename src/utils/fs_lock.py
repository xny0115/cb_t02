from pathlib import Path
import os
import psutil
import errno
import contextlib

@contextlib.contextmanager
def pid_lock(path: Path):
    path = Path(path)
    try:
        if path.exists():
            pid = int(path.read_text().strip())
            if psutil.pid_exists(pid):
                raise RuntimeError(f"Training already running under PID {pid}")
        path.write_text(str(os.getpid()))
        yield
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
