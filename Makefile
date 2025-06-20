deps:
	python setup.py
	node setup_js_deps.py


test:
	pytest -q
	pytest --cov=src -q
	npm test -- --coverage
