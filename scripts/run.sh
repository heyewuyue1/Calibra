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
    --database stack \
    --benchmark STACK \
    --max-retry 1 \
    --repeats 40

echo "=== Step 3: stop offline_train_server ==="
kill $OFFLINE_PID
wait $OFFLINE_PID 2>/dev/null || true

echo "=== Step 4: merge leap samples ==="
python Calibra/src/merge_leap_sample.py /home/hejiahao/Calibra/data/STACK/STACK_40it.pt /home/hejiahao/Calibra/data/STACK_leap.pt

echo "=== Step 5: run train ==="
python Calibra/src/train.py

echo "=== Step 6: start test_server ==="
python Calibra/src/test_server.py &
TEST_SERVER_PID=$!
echo "test_server pid = $TEST_SERVER_PID"

sleep 3

echo "=== Step 7: run test ==="
python Calibra/src/test.py \
    --method test \
    --database stack \
    --benchmark STACK \
    --max-retry 10 \
    --repeats 1 \
    --save-latency

echo "=== Step 8: stop test_server ==="
kill $TEST_SERVER_PID
wait $TEST_SERVER_PID 2>/dev/null || true


echo "=== Step 9: generate results ==="
python Calibra/src/plots/comp2json.py \
  Calibra/results/baselines/STACK_cbo.json \
  Calibra/results/STACK_40its.json

python Calibra/src/plots/comp2json.py \
  Calibra/results/baselines/STACK_leap.json \
  Calibra/results/STACK_40its.json

echo "=== ALL DONE ==="
