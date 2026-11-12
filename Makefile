# Thin wrapper over reproduce.sh, so `make` does the obvious thing.
# reproduce.sh holds the real dependency order; this file only names the tiers.

SHELL := /bin/bash
PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help recompute artifacts train figures all test env checksums clean clean-pyc

help:  ## show this help
	@echo "CapsNet / Entropy Bridge -- reproduction targets"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Start with 'make recompute': no GPU, minutes; the parameter-count check reads models/."
	@echo "Checkpoints (~2.4 GB) are distributed separately -- see README.md,"
	@echo "and check a download against models/CHECKSUMS.sha256."

recompute:  ## re-derive the paper's statistics from committed data (no GPU) -- START HERE
	@./reproduce.sh --recompute

artifacts:  ## regenerate all result artifacts from the checkpoints (GPU, hours)
	@./reproduce.sh --artifacts

train:  ## retrain every model from scratch (GPU, days)
	@./reproduce.sh --train

figures:  ## regenerate the figures (checkpoints; GPU optional)
	@./reproduce.sh --figures

all:  ## train, regenerate artifacts and figures, then check the statistics against that run
	@./reproduce.sh --all

test:  ## run the tests (need models/ and ./data; skipped with the reason otherwise)
	@$(PYTHON) -B -m pytest tests

env:  ## print the environment this run would use, next to the pinned one
	@echo "-- running --"
	@$(PYTHON) -c "import sys,torch,platform; \
	  print(f'  python  {platform.python_version()}'); \
	  print(f'  torch   {torch.__version__}'); \
	  print(f'  cuda    {torch.version.cuda} (available={torch.cuda.is_available()})')"
	@echo "-- pinned in requirements.txt --"
	@grep -E '^(torch|torchvision|numpy|scipy|statsmodels)==' requirements.txt | sed 's/^/  /'

checksums:  ## regenerate models/CHECKSUMS.sha256 from the checkpoints on disk
	@./train/checksum_models.sh

# Build products only. Nothing under results/ is touched: those are the canonical
# artifacts the paper reports, and regenerating them needs checkpoints and a GPU.
clean:  ## remove generated build products (__pycache__)
	@$(MAKE) --no-print-directory clean-pyc

clean-pyc:  ## remove __pycache__ trees
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null; true
	@echo "removed __pycache__"
