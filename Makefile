# Replication entry points (tier 1: exact reproduction from the cache; no API keys, no GPU).
# See replication/README.md.
PY ?= python3
IMAGE ?= spt-replication

.PHONY: replicate replicate-fast report env docker-build docker-replicate docker-replicate-fast clean-replication

## full mode: recompute every simulation unit (about 16-17 h on 4 cores, estimated), then everything fast mode does
replicate:
	$(PY) replication/replicate.py --mode full

## fast mode: fetch the verified simulation units, rerun every analysis, recompute a sample of units (about 55 min)
replicate-fast:
	$(PY) replication/replicate.py --mode fast

## rebuild replication/out/REPORT.md from the last run
report:
	$(PY) replication/replicate.py --report-only

## pinned local environment (Python 3.13 required)
env:
	$(PY) -m venv .venv && .venv/bin/pip install --require-hashes -r replication/requirements.lock
	@echo "now run: make replicate-fast PY=.venv/bin/python"

docker-build:
	docker build -t $(IMAGE) .

docker-replicate-fast: docker-build
	mkdir -p replication/out && docker run --rm -e HF_TOKEN -v "$(CURDIR)/replication/out:/repo/replication/out" $(IMAGE) python replication/replicate.py --mode fast

docker-replicate: docker-build
	mkdir -p replication/out && docker run --rm -e HF_TOKEN -v "$(CURDIR)/replication/out:/repo/replication/out" $(IMAGE) python replication/replicate.py --mode full

## remove replication outputs (the checkout itself is restored with: git checkout -- results figures data)
clean-replication:
	rm -rf replication/out
