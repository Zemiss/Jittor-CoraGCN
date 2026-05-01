.PHONY: setup train validate package clean

PYTHON ?= python3
CONFIG ?= ../configs/default.json
SEEDS ?= 13

setup:
	$(PYTHON) -m pip install -r requirements.txt

train:
	cd release && JITTOR_USE_CUDA=0 $(PYTHON) gcn.py --config $(CONFIG) --seeds $(SEEDS)

validate:
	$(PYTHON) scripts/validate_submission.py

package:
	$(PYTHON) scripts/package_submission.py

clean:
	rm -rf outputs release/outputs .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
