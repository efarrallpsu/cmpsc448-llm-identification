#!/usr/bin/env bash
# Extended retraining: 15 epochs, early-stopping patience 3.
# Reruns resume automatically (skip if result JSON already exists).
set -eo pipefail
mkdir -p logs

run() {
  local tag="$1"; shift
  if [ -f "results/result_${tag}.json" ]; then
    echo "=== ${tag} (already done, skipping) ==="
    return
  fi
  echo "=== ${tag} ==="
  venv/bin/python -u src/train.py "$@" --tag "$tag" --epochs 15 --patience 3 \
    2>&1 | tee "logs/extended_${tag}.log"
}

run input_cnn    --input input    --model cnn
run output_cnn   --input output   --model cnn
run combined_cnn --input combined --model cnn
run input_lstm    --input input    --model lstm
run output_lstm   --input output   --model lstm
run combined_lstm --input combined --model lstm

for exp in in_code in_math in_creative in_general noncreative_to_creative code_to_math general_to_code; do
  run "rq3_${exp}" --input output --model cnn --data_dir "data/processed/rq3/${exp}"
done

run rq4_len128_output_cnn --input output --model cnn --max_len 128

venv/bin/python src/summarize.py 2>&1 | tee logs/extended_summary.log
venv/bin/python src/make_figures.py 2>&1 | tee logs/extended_figures.log
touch logs/EXTENDED_DONE
