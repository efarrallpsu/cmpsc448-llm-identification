#!/usr/bin/env bash
# Wait for the main sweep, then run RQ3 transfer experiments and the
# RQ4 length-ablation run, and finally regenerate all figures.
set -e
mkdir -p logs
while [ ! -f logs/SWEEP_DONE ]; do sleep 30; done

for exp in in_code in_math in_creative in_general noncreative_to_creative code_to_math general_to_code; do
  echo "=== RQ3 $exp ==="
  venv/bin/python -u src/train.py --input output --model cnn \
    --data_dir "data/processed/rq3/$exp" --tag "rq3_$exp" \
    2>&1 | tee "logs/rq3_${exp}.log"
done

echo "=== RQ4 length ablation (max_len=128) ==="
venv/bin/python -u src/train.py --input output --model cnn --max_len 128 \
  --tag rq4_len128_output_cnn 2>&1 | tee logs/rq4_len128_output_cnn.log

venv/bin/python src/make_figures.py 2>&1 | tee logs/figures.log
touch logs/EXTRA_DONE
