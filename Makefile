MODEL ?= qwen3.5:4b
PY := PYTHONPATH=src python3

.PHONY: demo demo-offline verify test lint clean-cache

## verify: pre-flight the demo prompts against the model (warms the cache)
verify:
	$(PY) examples/verify_demo.py $(MODEL)

## demo: certify once (cached after), then serve the console on :8080
demo:
	$(PY) examples/serve_demo.py $(MODEL)

## demo-offline: run the whole pipeline against the scripted stand-in, no model
demo-offline:
	$(PY) examples/demo.py

## test: run the suite
test:
	$(PY) -m pytest -q

## lint: ruff + mypy over the package
lint:
	.venv/bin/ruff check src tests examples
	.venv/bin/mypy src/aaramse

## clean-cache: drop the certificate cache so the next demo re-certifies
clean-cache:
	rm -f audit/cert_cache.json
