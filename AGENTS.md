# Repository Guidelines

- 在试图做事情的时候，比如编辑文件等时候，不要再尝试什么sandbox了，直接修改源文件。
- 项目的Python版本是3.8.20，一些语法注意兼容一下否则Pylance会报错

## Project Structure & Module Organization
`src/` contains the application code: training entrypoints (`train.py`, `bootstrap.py`), FastAPI services (`offline_train_server.py`, `test_server.py`, `bootstrap_server.py`), and shared packages under `models/`, `preprocessor/`, `tree_conv/`, `utils/`, and `plots/`. `benchmark/` stores SQL workloads such as `STACK/` and `TPC-H/`. `conf/` holds Spark property files for each mode. `data/`, `results/`, and `logs/` contain local artifacts, trained models, and experiment output. `scripts/` wraps common end-to-end flows, and `jar/` holds the Spark-side runtime JAR.

## Build, Test, and Development Commands
Run commands from the repository root unless a script assumes the parent directory.

- `bash scripts/start_all.sh`: start the local Spark cluster and history server.
- `python src/offline_train_server.py`: start the data-collection service on port `10533`.
- `python src/test.py --method offline_train --database stack --benchmark STACK --max-retry 1 --repeats 40`: execute benchmark SQL and collect latency samples.
- `python src/train.py`: train the cost model from the dataset configured in `src/config.py`.
- `python src/test_server.py`: serve the trained model for inference.
- `bash scripts/run.sh`: run the full collection, training, and evaluation pipeline.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for functions, variables, and modules, PascalCase for classes and Pydantic models. Keep new code close to the current script-oriented structure; repository entrypoints are plain Python files rather than console wrappers. No formatter or linter is configured in-tree, so keep imports readable, prefer small functions, and update shared constants in `src/config.py` instead of scattering paths.

## Testing Guidelines
This repository uses workflow validation instead of a dedicated `pytest` suite. Treat `src/test.py` as the benchmark runner and `src/test_server.py` as the inference smoke test. After changing preprocessing, model code, or request schemas, rerun the affected pipeline and confirm updated JSON or plot outputs under `results/`. Include the exact command used and the benchmark name in your change notes.

## Commit & Pull Request Guidelines
Recent commits use short date-based subjects such as `20260325`. Keep that pattern, optionally adding a brief scope suffix, for example `20260327 train logging`. Pull requests should state the benchmark or dataset affected, list config changes, summarize commands run, and attach before/after metrics or result files when model behavior changes.

## Configuration Notes
Paths are currently hard-coded to `/home/hejiahao/Calibra` and local Spark installations in `src/config.py` and `scripts/`. Update those paths deliberately and mention any environment assumptions in the PR. Avoid committing large generated data, logs, or Spark temp files unless they are intentional fixtures.
