# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment and assumptions
- Python version is 3.8.20.
- Many paths are hard-coded to `/home/hejiahao/Calibra` in `src/config.py` and to a local Spark install in `scripts/start_all.sh`; check those before changing environments.
- `scripts/run.sh` uses paths like `Calibra/src/...`, so it is intended to be run from the parent directory of the repo, not from the repo root.
- There is no in-repo linter or pytest suite configured.

## Common commands
- Start local Spark services: `bash scripts/start_all.sh`
- Start offline data-collection server: `python src/offline_train_server.py`
- Start bootstrap pair-collection server: `python src/bootstrap_server.py`
- Start inference server with a trained model: `python src/test_server.py`
- Collect benchmark latencies / replay a workload: `python src/test.py --method offline_train --database stack --benchmark STACK --max-retry 1 --repeats 40`
- Evaluate with the served model and save latencies: `python src/test.py --method test --database stack --benchmark STACK --max-retry 10 --repeats 1 --save-latency`
- Train the pointwise latency model: `python src/train.py`
- Train the bootstrap pairwise model: `python src/bootstrap.py`
- Run the end-to-end offline pipeline (from the parent directory of this repo): `bash Calibra/scripts/run.sh`
- Generate comparison JSON from baseline and experiment outputs: `python src/plots/comp2json.py results/baselines/STACK_cbo.json results/STACK_40its.json`

## Running a single query manually
`src/test.py` always iterates over every SQL file in a benchmark directory. To run one SQL file, use `spark-sql` directly with the same flags that `src/test.py` builds:

```bash
spark-sql \
  --database stack \
  -f benchmark/STACK/<query>.sql \
  --name manual-<query> \
  --properties-file conf/offline_train.conf
```

Swap the database, SQL path, and properties file (`offline_train.conf`, `test.conf`, `bootstrap.conf`, etc.) to match the mode you want.

## High-level architecture
The repository is a Spark SQL plan-cost modeling pipeline built around three stages:

1. **Collect plan/execution data from Spark**
   - FastAPI servers in `src/offline_train_server.py`, `src/bootstrap_server.py`, and `src/test_server.py` expose `/cost` and related endpoints on port `10533`.
   - Request/response payloads are defined in `src/request_models.py`.
   - `src/test.py` drives benchmark SQL execution through `spark-sql`, using the selected properties file under `conf/` so Spark talks to the active server.

2. **Turn Spark plans into model features**
   - `src/preprocessor/sparkplanpreprocessor.py` parses logical plans, physical plans, and query-stage plans into binary trees of `Node` objects.
   - The preprocessor extracts operator types, table usage, estimated card/size, AQE/query-stage information, and scan predicates.
   - `src/models/encoder.py` converts trees into nested feature tuples. `UnifiedFeatureEncoder` is the single encoder used by current training/inference paths, and `TrainConfig.enable_predicate_encoding` controls whether predicate features are included.
   - `src/utils/util.py` flattens tree-structured features into the schedule-aware tensors consumed by TreeLRU.

3. **Train or serve the neural cost model**
   - `src/models/TreeLRUNet.py` defines the main model as stacked TreeLRU layers plus pooling and linear heads.
   - `src/bootstrap.py` trains a pairwise bootstrap model from plan pairs collected by `bootstrap_server.py`.
   - `src/train.py` is the single training entrypoint for the pointwise latency model; it reads `TrainConfig.enable_predicate_encoding` and writes checkpoints/log artifacts through the dynamic `TrainConfig` path helpers.
   - `src/test_server.py` loads the trained checkpoint and returns predicted costs for candidate plans.

## Important repository structure
- `src/config.py` is the central place for dataset paths, checkpoint paths, benchmark selection, and environment-specific constants.
- `benchmark/` stores SQL workloads grouped by benchmark.
- `conf/` contains Spark properties for each collection/evaluation mode.
- `data/` stores serialized training datasets and model checkpoints.
- `results/` stores benchmark latency outputs and generated comparison artifacts.
- `scripts/` contains orchestration scripts; `scripts/run.sh` is the main offline pipeline wrapper.
- `jar/` contains the Spark-side runtime integration.

## Working conventions that matter here
- Keep changes compatible with Python 3.8 syntax.
- Prefer updating constants and paths in `src/config.py` instead of scattering environment-specific values.
- `offline_train_server.py` and `test_server.py` both bind to port `10533`; only one of these servers should be active at a time.
- Current commit history uses short date-based subjects such as `20260325`.
