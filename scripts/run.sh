#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PARENT="$(dirname "$REPO_ROOT")"
cd "$PROJECT_PARENT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="${CALIBRA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BENCHMARK="STACK"
DATABASE="stack"
TABLE_KEY="stack"
RAW_REPEATS=40
RAW_MAX_RETRY=1
EVAL_REPEATS=1
EVAL_MAX_RETRY=10
PREDICATE_ENCODING=true
export CALIBRA_RUN_ID="$RUN_ID"
export CALIBRA_BENCHMARK="$BENCHMARK"

load_artifact_paths() {
  python - "$BENCHMARK" "$RUN_ID" <<'PY'
import sys
from config import get_pretrained_run_spec, get_run_artifacts

benchmark = sys.argv[1]
run_id = sys.argv[2]
artifacts = get_run_artifacts(benchmark, run_id)
spec = get_pretrained_run_spec(benchmark)
print(artifacts.raw_training_data_path)
print(artifacts.model_path)
print(artifacts.metrics_path)
print(artifacts.pointwise_tensorboard_dir)
print(artifacts.latency_path)
print(artifacts.manifest_path)
print(str(artifacts.conf_dir))
print(str(artifacts.log_dir))
print(artifacts.log_artifact_path("workflow.log"))
print(artifacts.comparison_plot_path("cbo"))
print(artifacts.comparison_plot_path("leap"))
print(spec.baseline_cbo_path)
print(spec.baseline_leap_path)
PY
}

mapfile -t PATHS < <(load_artifact_paths)
RAW_DATA_PATH="${PATHS[0]}"
MODEL_PATH="${PATHS[1]}"
METRICS_PATH="${PATHS[2]}"
TENSORBOARD_DIR="${PATHS[3]}"
LATENCY_PATH="${PATHS[4]}"
MANIFEST_PATH="${PATHS[5]}"
CONF_DIR="${PATHS[6]}"
LOG_DIR="${PATHS[7]}"
WORKFLOW_LOG_PATH="${PATHS[8]}"
CBO_PLOT_PATH="${PATHS[9]}"
LEAP_PLOT_PATH="${PATHS[10]}"
BASELINE_CBO="${PATHS[11]}"
BASELINE_LEAP="${PATHS[12]}"

python - "$BENCHMARK" "$RUN_ID" "$DATABASE" "$TABLE_KEY" "$RAW_DATA_PATH" "$MODEL_PATH" "$METRICS_PATH" "$LATENCY_PATH" "$BASELINE_CBO" "$BASELINE_LEAP" "$PREDICATE_ENCODING" "$RAW_MAX_RETRY" "$RAW_REPEATS" "$EVAL_MAX_RETRY" "$EVAL_REPEATS" "$CONF_DIR" "$LOG_DIR" "$WORKFLOW_LOG_PATH" <<'PY'
import subprocess
import sys
from config import ensure_dir, get_run_artifacts, update_manifest

(
    benchmark,
    run_id,
    database,
    table_key,
    raw_data_path,
    model_path,
    metrics_path,
    latency_path,
    baseline_cbo,
    baseline_leap,
    predicate_encoding,
    raw_max_retry,
    raw_repeats,
    eval_max_retry,
    eval_repeats,
    conf_dir,
    log_dir,
    workflow_log_path,
) = sys.argv[1:19]
artifacts = get_run_artifacts(benchmark, run_id)
git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
ensure_dir(conf_dir)
ensure_dir(log_dir)
update_manifest(
    artifacts.manifest_path,
    {
        **artifacts.manifest_defaults(),
        "workflow": {
            "benchmark": benchmark,
            "database": database,
            "table_key": table_key,
            "run_id": run_id,
            "predicate_encoding": predicate_encoding == "true",
            "raw_max_retry": int(raw_max_retry),
            "raw_repeats": int(raw_repeats),
            "eval_max_retry": int(eval_max_retry),
            "eval_repeats": int(eval_repeats),
            "raw_data_path": raw_data_path,
            "model_path": model_path,
            "metrics_path": metrics_path,
            "latency_path": latency_path,
            "baseline_cbo_path": baseline_cbo,
            "baseline_leap_path": baseline_leap,
            "log_path": workflow_log_path,
            "git_commit": git_commit,
        },
    },
)
print(artifacts.manifest_path)
PY

OFFLINE_PID=""
TEST_SERVER_PID=""
SERVER_HOST="localhost"
SERVER_PORT=10533
SERVER_READY_TIMEOUT=60
SERVER_STOP_TIMEOUT=20

mkdir -p "$CONF_DIR" "$LOG_DIR"
: > "$WORKFLOW_LOG_PATH"
exec > >(tee -a "$WORKFLOW_LOG_PATH")
exec 2>&1

port_is_open() {
  local port="$1"
  python - "$SERVER_HOST" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

try:
    with socket.create_connection((host, port), timeout=1):
        sys.exit(0)
except OSError:
    sys.exit(1)
PY
}

server_is_ready() {
  local port="$1"
  python - "$SERVER_HOST" "$port" <<'PY'
import sys
import urllib.request

host = sys.argv[1]
port = int(sys.argv[2])
url = f"http://{host}:{port}/openapi.json"

try:
    with urllib.request.urlopen(url, timeout=1) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

describe_port_owner() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN || true
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp "( sport = :$port )" || true
  else
    echo "Unable to inspect port $port: neither lsof nor ss is available."
  fi
}

ensure_port_free() {
  local port="$1"
  if port_is_open "$port"; then
    echo "Port $port is already in use before starting a server." >&2
    describe_port_owner "$port" >&2
    return 1
  fi
}

wait_for_server_ready() {
  local pid="$1"
  local port="$2"
  local name="$3"
  local elapsed=0

  while (( elapsed < SERVER_READY_TIMEOUT )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      echo "$name exited before becoming ready on port $port." >&2
      return 1
    fi

    if server_is_ready "$port"; then
      return 0
    fi

    sleep 1
    elapsed=$((elapsed + 1))
  done

  echo "$name did not become ready on port $port within ${SERVER_READY_TIMEOUT}s." >&2
  describe_port_owner "$port" >&2
  return 1
}

wait_for_port_release() {
  local port="$1"
  local name="$2"
  local elapsed=0

  while (( elapsed < SERVER_STOP_TIMEOUT )); do
    if ! port_is_open "$port"; then
      return 0
    fi

    sleep 1
    elapsed=$((elapsed + 1))
  done

  echo "Port $port is still in use after stopping $name." >&2
  describe_port_owner "$port" >&2
  return 1
}

start_server() {
  local name="$1"
  local pid_var="$2"
  shift 2

  ensure_port_free "$SERVER_PORT"
  setsid "$@" --port "$SERVER_PORT" &
  local pid=$!
  printf -v "$pid_var" '%s' "$pid"
  echo "$name pid = $pid"
  wait_for_server_ready "$pid" "$SERVER_PORT" "$name"
}

stop_server() {
  local name="$1"
  local pid="$2"
  local elapsed=0

  if [[ -z "$pid" ]]; then
    return 0
  fi

  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true

    while kill -0 "$pid" 2>/dev/null; do
      if (( elapsed >= SERVER_STOP_TIMEOUT )); then
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        break
      fi

      sleep 1
      elapsed=$((elapsed + 1))
    done

    wait "$pid" 2>/dev/null || true
  fi

  wait_for_port_release "$SERVER_PORT" "$name"
}

cleanup() {
  stop_server "test_server" "$TEST_SERVER_PID" || true
  TEST_SERVER_PID=""
  stop_server "offline_train_server" "$OFFLINE_PID" || true
  OFFLINE_PID=""
}
trap cleanup EXIT

echo "=== Run ID: $RUN_ID ==="
echo "=== Manifest: $MANIFEST_PATH ==="
echo "=== Workflow log: $WORKFLOW_LOG_PATH ==="

echo "=== Step 1: start offline_train_server ==="
start_server \
  "offline_train_server" \
  OFFLINE_PID \
  python "$REPO_ROOT/src/offline_train_server.py" \
  --benchmark "$BENCHMARK" \
  --run-id "$RUN_ID"

echo "=== Step 2: collect offline training data ==="
python "$REPO_ROOT/src/test.py" \
  --method offline_train \
  --database "$DATABASE" \
  --benchmark "$BENCHMARK" \
  --run-id "$RUN_ID" \
  --max-retry "$RAW_MAX_RETRY" \
  --repeats "$RAW_REPEATS"

echo "=== Step 3: stop offline_train_server ==="
stop_server "offline_train_server" "$OFFLINE_PID"
OFFLINE_PID=""

echo "=== Step 4: run train ==="
if [[ "$PREDICATE_ENCODING" == "true" ]]; then
  python "$REPO_ROOT/src/train.py" \
    --benchmark "$BENCHMARK" \
    --run-id "$RUN_ID" \
    --data-path "$RAW_DATA_PATH" \
    --model-save-path "$MODEL_PATH" \
    --metrics-path "$METRICS_PATH" \
    --tensorboard-dir "$TENSORBOARD_DIR" \
    --predicate-encoding
else
  python "$REPO_ROOT/src/train.py" \
    --benchmark "$BENCHMARK" \
    --run-id "$RUN_ID" \
    --data-path "$RAW_DATA_PATH" \
    --model-save-path "$MODEL_PATH" \
    --metrics-path "$METRICS_PATH" \
    --tensorboard-dir "$TENSORBOARD_DIR" \
    --no-predicate-encoding
fi

echo "=== Step 5: start test_server ==="
if [[ "$PREDICATE_ENCODING" == "true" ]]; then
  start_server \
    "test_server" \
    TEST_SERVER_PID \
      python "$REPO_ROOT/src/test_server.py" \
    --benchmark "$BENCHMARK" \
    --database "$DATABASE" \
    --table-key "$TABLE_KEY" \
    --run-id "$RUN_ID" \
    --model-path "$MODEL_PATH" \
    --predicate-encoding
else
  start_server \
    "test_server" \
    TEST_SERVER_PID \
      python "$REPO_ROOT/src/test_server.py" \
    --benchmark "$BENCHMARK" \
    --database "$DATABASE" \
    --table-key "$TABLE_KEY" \
    --run-id "$RUN_ID" \
    --model-path "$MODEL_PATH" \
    --no-predicate-encoding
fi

echo "=== Step 6: run evaluation ==="
python "$REPO_ROOT/src/test.py" \
  --method test \
  --database "$DATABASE" \
  --benchmark "$BENCHMARK" \
  --run-id "$RUN_ID" \
  --max-retry "$EVAL_MAX_RETRY" \
  --repeats "$EVAL_REPEATS" \
  --save-latency \
  --save-path "$LATENCY_PATH"

echo "=== Step 7: stop test_server ==="
stop_server "test_server" "$TEST_SERVER_PID"
TEST_SERVER_PID=""

echo "=== Step 8: generate comparison plots ==="
python "$REPO_ROOT/src/plots/comp2json.py" \
  "$BASELINE_CBO" \
  "$LATENCY_PATH" \
  --output-path "$CBO_PLOT_PATH"

python "$REPO_ROOT/src/plots/comp2json.py" \
  "$BASELINE_LEAP" \
  "$LATENCY_PATH" \
  --output-path "$LEAP_PLOT_PATH"

echo "=== ALL DONE ==="
echo "Artifacts stored under: $REPO_ROOT/artifacts/$BENCHMARK/runs/$RUN_ID"
