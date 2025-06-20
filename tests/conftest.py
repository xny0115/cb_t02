import importlib.util, pytest

torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:
    pytest.skip("PyTorch not available in this environment", allow_module_level=True)
