.PHONY: install test smoke compile

install:
	pip install -e ".[hf,dev]"

compile:
	python -m compileall -q scenerepair run.py

test:
	python -m pytest -q

smoke:
	python scripts/make_toy_data.py
	python -m scenerepair.cli --config configs/mock_smoke.yaml --task all
