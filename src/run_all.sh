#!/usr/bin/env bash
# Train all RQ1/RQ2 configurations: 3 input types x 2 model types.
set -e
mkdir -p logs
for input in input output combined; do
  for model in cnn lstm; do
    if [ -f "results/result_${input}_${model}.json" ]; then
      echo "=== $input / $model (already done, skipping) ==="
      continue
    fi
    echo "=== $input / $model ==="
    venv/bin/python -u src/train.py --input "$input" --model "$model" \
      2>&1 | tee "logs/train_${input}_${model}.log"
  done
done
echo "ALL DONE"
touch logs/SWEEP_DONE
