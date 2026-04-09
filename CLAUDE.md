# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this project.

## What this repo is
- Calibra is a Spark SQL plan-cost modeling pipeline plus Spark-side runtime integration.
- The Python repo at `/home/hejiahao/Calibra` handles data collection, feature extraction, model training, inference serving, and experiment artifacts.
- Spark-side Scala integration that talks to the Python servers lives in `RuntimeCost/` in this repo.
- A separate Spark 3.5.4 source tree with broader Calibra planner changes lives at `/home/hejiahao/spark-3.5.4-calibra`.

## Environment and assumptions
- Python version is 3.8.20.
- The project's Python environment uses the `aqora` conda environment.
- There is no in-repo linter or pytest suite configured.
- Many paths are hard-coded to local machine paths. Check these before changing environments:
  - `src/config.py`
  - `conf/offline_train.conf`
  - `conf/test.conf`
  - `scripts/start_all.sh`
  - `scripts/clean_all.sh`
- `offline_train_server.py` and `test_server.py` both bind to port `10533`; only one of them should be active at a time.
- `scripts/clean_all.sh` is destructive: it removes Spark logs, tmp, and work directories under the local Spark installs.

## Active workflow
The current supported workflow is run-centric and writes outputs under `artifacts/<BENCHMARK>/runs/<RUN_ID>/`.

A typical run is:
1. Start `src/offline_train_server.py`
2. Run `src/test.py --method offline_train ...` to collect labeled plans into `dataset/raw_train.pt`
3. Merge the collected data with retained LEAP samples
4. Run `src/train.py` to train the pointwise model
5. Start `src/test_server.py` with the trained checkpoint
6. Run `src/test.py --method test ... --save-latency` to evaluate
7. Run `src/plots/comp2json.py` to generate comparison plots against baselines

The canonical path layout for new runs is defined by `RunArtifacts` in `src/config.py`.

## Important commands
Run these from `/home/hejiahao/Calibra` unless the command itself says otherwise.

- Start local Spark services: `bash scripts/start_all.sh`
- Clean local Spark logs/tmp/work dirs: `bash scripts/clean_all.sh`
- Start offline data-collection server: `conda run -n aqora python src/offline_train_server.py --benchmark <BENCHMARK> --run-id <RUN_ID>`
- Start inference server with a trained model: `conda run -n aqora python src/test_server.py --benchmark <BENCHMARK> --run-id <RUN_ID> --database <DATABASE> --table-key <TABLE_KEY> --model-path <MODEL_PATH> --predicate-encoding`
- Collect offline training data: `conda run -n aqora python src/test.py --method offline_train --database <DATABASE> --benchmark <BENCHMARK> --run-id <RUN_ID> --max-retry 1 --repeats <N>`
- Evaluate with the served model and save latencies: `conda run -n aqora python src/test.py --method test --database <DATABASE> --benchmark <BENCHMARK> --run-id <RUN_ID> --max-retry 10 --repeats 1 --save-latency --save-path <LATENCY_JSON>`
- Train the pointwise latency model from scratch: `conda run -n aqora python src/train.py --benchmark <BENCHMARK> --run-id <RUN_ID> --model-save-path <MODEL_PATH> --metrics-path <METRICS_CSV> --tensorboard-dir <TENSORBOARD_DIR> --predicate-encoding`
- Generate comparison plots from baseline and experiment outputs: `conda run -n aqora python src/plots/comp2json.py <baseline.json> <latency.json> --output-path <plot.png>`

## About `scripts/run.sh`
- `scripts/run.sh` is the main end-to-end wrapper, but it is currently STACK-oriented:
  - it hardcodes `BENCHMARK="STACK"`
  - it uses `STACK` baselines
  - it merges with `data/STACK_leap.pt`
- It self-locates and changes into the project parent internally, so it no longer needs to be launched from the repo parent manually.
- When adapting the wrapper for JOB or TPC-H, update the benchmark/database/table-key and baseline/LEAP paths consistently.

## LEAP merge step caveat
- The current workflow expects a merge step between collected raw data and retained LEAP samples.
- `scripts/run.sh` invokes `src/merge_leap_sample.py`, but in the current tree only the compiled helper `src/__pycache__/merge_leap_sample.cpython-38.pyc` is present.
- If a run depends on that merge step, verify whether the `.pyc` helper is the intended executable in this workspace before changing the workflow.

## Running a single query manually
`src/test.py` always iterates over every SQL file in a benchmark directory. To run one SQL file, use `spark-sql` directly with the same flags that `src/test.py` builds:

```bash
spark-sql \
  --database stack \
  -f benchmark/STACK/<query>.sql \
  --name manual-<query> \
  --properties-file conf/offline_train.conf
```

Use the properties file that matches the active mode, typically `offline_train.conf` for collection or `test.conf` for evaluation.

## High-level architecture
### 1. Collect plan and execution data from Spark
- `src/offline_train_server.py` receives `/cost` requests to queue candidate plans and `/register` requests to attach execution labels and save `raw_train.pt`.
- `src/test.py` drives benchmark SQL execution through `spark-sql`.
- Request and response payloads are defined in `src/request_models.py`.
- Collection uses `conf/offline_train.conf`, which loads the RuntimeCost jar, `cn.edu.ruc.QueryRegisterListener`, and `cn.edu.ruc.LogicalPlanSenderInjector`.

### 2. Turn Spark plans into model features
- `src/preprocessor/sparkplanpreprocessor.py` parses logical plans, physical plans, and query-stage plans into binary trees of `Node` objects.
- The preprocessor extracts operator types, table usage, estimated card/size, AQE/query-stage information, and scan predicates.
- `src/models/encoder.py` contains `UnifiedFeatureEncoder`, which is the active encoder used by current training and inference paths.
- `TrainConfig.enable_predicate_encoding` in `src/config.py` controls predicate feature usage.
- `src/utils/util.py` flattens tree-structured features into the tensors consumed by TreeLRU.

### 3. Train or serve the neural cost model
- `src/models/TreeLRUNet.py` defines the main TreeLRU-based latency model.
- `src/train.py` is the supported training entrypoint for the pointwise latency model.
- `src/test_server.py` loads a trained checkpoint and returns predicted costs for candidate plans.
- Evaluation uses `conf/test.conf`, which enables `cn.edu.ruc.RuntimeCostEvaluator`, keeps `LogicalPlanSenderInjector`, and points Spark to `http://localhost:10533`.

## RuntimeCost directory
`RuntimeCost/` is the editable Spark-side Scala integration packaged into `jar/runtimecost_2.12-0.1.0-SNAPSHOT.jar`.

The key files are:
- `RuntimeCost/src/main/scala/WebUtils.scala` — HTTP client used to talk to the Python servers
- `RuntimeCost/src/main/scala/LogicalPlanSender.scala` — sends logical plans and partially executed plans to `/cost`
- `RuntimeCost/src/main/scala/QueryRegisterListener.scala` — registers final execution times with `/register`
- `RuntimeCost/src/main/scala/RuntimeCostEvaluator.scala` — AQE custom cost evaluator for test-time plan selection
- `RuntimeCost/src/main/scala/SubquerySelection.scala` — logical join-order selection that uses Python-side costs
- `RuntimeCost/build.sbt` — Scala build definition for the RuntimeCost jar

When collection or evaluation behavior changes, check both the Python servers and the RuntimeCost Scala code.

## External Spark source tree
A separate Spark source tree with Calibra-specific planner integration lives at:
- `/home/hejiahao/spark-3.5.4-calibra`

Use it when you need to understand or modify Spark-side planner behavior beyond the RuntimeCost jar.

The most relevant Spark-side files are:
- `/home/hejiahao/spark-3.5.4-calibra/sql/catalyst/src/main/scala/org/apache/spark/sql/util/CalibraClient.scala`
- `/home/hejiahao/spark-3.5.4-calibra/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/optimizer/CostBasedJoinReorder.scala`
- `/home/hejiahao/spark-3.5.4-calibra/sql/core/src/main/scala/org/apache/spark/sql/execution/SparkPlanner.scala`
- `/home/hejiahao/spark-3.5.4-calibra/sql/core/src/main/scala/org/apache/spark/sql/execution/SparkStrategies.scala`
- `/home/hejiahao/spark-3.5.4-calibra/sql/core/src/main/scala/org/apache/spark/sql/execution/QueryExecution.scala`

Important distinction:
- `/home/hejiahao/spark-3.5.4-calibra` is the source/build tree with Calibra patches.
- `/home/hejiahao/spark-3.5.4-bin-hadoop3` is the installed runtime Spark distribution used by `spark-sql`, `scripts/start_all.sh`, and the actual experiment runs.

Do not confuse source changes with the active runtime. If behavior differs, verify whether rebuilt jars from the source tree have actually been copied into the installed Spark distribution.

## Important repository structure
- `src/config.py` — canonical benchmark specs, run IDs, artifact layout, and environment-specific constants
- `src/offline_train_server.py` — collection server for training data
- `src/test_server.py` — inference server for evaluation
- `src/test.py` — benchmark runner for collection and evaluation
- `src/train.py` — pointwise model training entrypoint
- `src/preprocessor/` — plan parsing and feature extraction
- `src/models/` — encoders and TreeLRU model definitions
- `conf/` — Spark property files for collection/evaluation
- `benchmark/` — SQL workloads grouped by benchmark
- `RuntimeCost/` — Scala integration packaged as the Calibra runtime jar
- `jar/` — built RuntimeCost jar dropped into the Python repo
- `artifacts/` — primary output location for new run datasets, models, metrics, latencies, and plots
- `results/` — retained baselines and curated comparison outputs
- `logs/` — date-stamped process logs written by the Python logger
- `data/` — retained datasets and pretrained/reference models, not the primary sink for new runs

## Artifacts and logs
For a current or recent run, inspect these first:
- `artifacts/<BENCHMARK>/runs/<RUN_ID>/manifest.json` — run metadata and canonical paths
- `artifacts/<BENCHMARK>/runs/<RUN_ID>/dataset/` — collected and merged datasets
- `artifacts/<BENCHMARK>/runs/<RUN_ID>/model/` — trained checkpoints
- `artifacts/<BENCHMARK>/runs/<RUN_ID>/train/` — metrics and TensorBoard outputs
- `artifacts/<BENCHMARK>/runs/<RUN_ID>/eval/` — saved latency JSON
- `artifacts/<BENCHMARK>/runs/<RUN_ID>/plots/` — comparison plots
- `logs/<YYYY-MM-DD>/` — wall-clock logs from `TEST`, `SERVER`, and `TRAIN`

Run directories may be partially populated while a workflow is still executing.

## Bootstrap artifacts
- `src/bootstrap.py`, `src/bootstrap_server.py`, and `conf/bootstrap.conf` are retained historical artifacts.
- Do not use them as the default path for current collection, training, or evaluation.
- Current supported training goes directly through `src/train.py` and the run-oriented workflow above.

## Working conventions that matter here
- Keep changes compatible with Python 3.8 syntax.
- Prefer updating constants and paths in `src/config.py` instead of scattering environment-specific values.
- Prefer `RunArtifacts` in `src/config.py` for new output paths instead of hardcoding result locations.
- Treat `artifacts/` as the default destination for new experiment outputs.
- Use `logs/<date>/...` when checking live progress; use `artifacts/.../manifest.json` when checking structured run state.
- Current commit messages are descriptive and not necessarily date-only subjects.
