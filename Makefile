# SWATTER — heavy work runs on the DGX. Cheap checks stay local.

# `make help` lists the targets. The split is the point: anything that touches
# the 1.05 GB connectome table or sweeps thousands of LIF trials goes through
# ./scripts/dgx.sh, and the heavy scripts themselves refuse to start locally
# unless SWATTER_ALLOW_LOCAL_HEAVY=1.

.PHONY: help
help:
	@echo "local (cheap):   stimuli suite-sizes lint test check clean-local"
	@echo "DGX (expensive): setup fetch extract loom-sweep ablations export status logs pull"

# --- local, cheap --------------------------------------------------------- #

.PHONY: stimuli
stimuli: ## Summarise the lab stimulus suite (no connectome data needed)
	.venv/bin/python offline/loom.py --summary

.PHONY: suite
suite: ## Write the full criterion sweep to docs/loom_suite.json
	@mkdir -p docs
	.venv/bin/python offline/loom.py --suite --save docs/loom_suite.json

.PHONY: manifest
manifest: ## Print the extracted escape-circuit counts from data/subgraph
	@.venv/bin/python -c "import json,sys; \
m=json.load(open('data/subgraph/manifest.json')); \
print('neurons', f\"{m['annotated_neurons']:,}\"); \
print('edges  ', f\"{m['edges']:,}\"); \
print('synapses', f\"{m['synapses']:,}\"); \
[print(f\"  {r['target']:<7} from_LPLC2={r['from_LPLC2']:>7,} from_LC4={r['from_LC4']:>7,} share={r.get('detector_share_of_incoming',0):.1%}\") for r in m['escape_circuit_core']]"

.PHONY: lint
lint: ## Byte-compile every module, so syntax errors surface without running jobs
	.venv/bin/python -m compileall -q offline scripts

.PHONY: test
test: ## Run the cheap unit tests (no connectome data required)
	.venv/bin/python -m unittest discover -s tests -v

.PHONY: check
check: lint test stimuli manifest ## Everything cheap, in one go

.PHONY: clean-local
clean-local: ## Remove caches. Never touches data/.
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache

# --- DGX, expensive ------------------------------------------------------- #

.PHONY: setup
setup: ## One-time: sync code to the DGX and build its venv
	./scripts/dgx.sh setup

.PHONY: fetch
fetch: ## DGX: download the MaleCNS tables (~1.1 GB)
	./scripts/dgx.sh run fetch_malecns --group core --columns

.PHONY: extract
extract: ## DGX: build the escape subgraph and count it
	./scripts/dgx.sh run extract_subgraph

.PHONY: loom-sweep
loom-sweep: ## DGX: run the four success criteria across the r/v sweep
	./scripts/dgx.sh run loom_sweep

.PHONY: ablations
ablations: ## DGX: LPLC2-silenced, GF-silenced, shuffled-weights, threshold-bot
	./scripts/dgx.sh run loom_sweep --ablations

.PHONY: export
export: ## DGX: freeze parameters and write the browser weight bundle
	./scripts/dgx.sh run export

.PHONY: status
status: ## What is running on the DGX
	./scripts/dgx.sh status

.PHONY: logs
logs: ## Tail a DGX job log: make logs JOB=extract_subgraph
	./scripts/dgx.sh logs $(JOB)

.PHONY: pull
pull: ## Fetch DGX results back into data/subgraph
	./scripts/dgx.sh pull
