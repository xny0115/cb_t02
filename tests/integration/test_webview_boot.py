import importlib
import threading
import time

def test_boot_no_recursion():
    app = importlib.import_module("run")
    t = threading.Thread(target=app.main, daemon=True)
    t.start()
    time.sleep(3)
    assert t.is_alive()
