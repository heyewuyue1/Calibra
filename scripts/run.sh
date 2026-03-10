#!/usr/bin/env bash
set -e

echo "=== Step 1: start offline_train_server ==="
python Calibra/src/offline_train_server.py &
OFFLINE_PID=$!
echo "offline_train_server pid = $OFFLINE_PID"

sleep 3

echo "=== Step 2: run test ==="
python Calibra/src/test.py \
    --method offline_train \
    --database tpch_sf100 \
    --benchmark TPC-H \
    --max-retry 1 \
    --repeats 40

echo "=== Step 3: stop offline_train_server ==="
kill $OFFLINE_PID
wait $OFFLINE_PID 2>/dev/null || true

echo "=== Step 4: run train ==="
python Calibra/src/train.py

echo "=== Step 5: start test_server ==="
python Calibra/src/test_server.py &
TEST_SERVER_PID=$!
echo "test_server pid = $TEST_SERVER_PID"

sleep 3

echo "=== Step 6: run test ==="
python Calibra/src/test.py \
    --method test \
    --database tpch_sf100 \
    --benchmark TPC-H \
    --max-retry 10 \
    --repeats 1 \
    --save-latency

echo "=== Step 7: stop test_server ==="
kill $TEST_SERVER_PID
wait $TEST_SERVER_PID 2>/dev/null || true


echo "=== Step 8: generate results ==="
python Calibra/src/plots/comp2json.py \
  Calibra/results/TPC-H_cbo.json \
  Calibra/results/TPC-H_l4k4r_40it.json

python Calibra/src/plots/comp2json.py \
  Calibra/results/TPC-H_leap.json \
  Calibra/results/TPC-H_l4k4r_40it.json

echo "=== ALL DONE ==="
