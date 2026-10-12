# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project 1 (individual). A 6-way text classifier that
predicts which LLM family wrote a response, using a Text-CNN and a BiLSTM.

## Data

Source: [`lmarena-ai/arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)
(CC-BY-4.0). Each raw row is a Chatbot Arena battle: one prompt answered by
two models. `src/build_dataset.py` downloads the shards and flattens them
into `(model, prompt, response)` samples.

Selected models (one per family): `gpt-4.1-2025-04-14`,
`claude-sonnet-4-20250514`, `gemini-2.5-pro`,
`qwen3-235b-a22b-no-thinking`, `deepseek-v3-0324`,
`llama-4-maverick-03-26-experimental`. Final dataset: 9000 balanced samples
(1500/model), split 70/15/15 by prompt-group hash so no prompt leaks across
splits.

## Layout

```
src/build_dataset.py   download + flatten LMArena shards
src/make_splits.py     model selection, balance, leakage-free splits
src/textdata.py        tokenizer, vocabulary, Dataset/collate
src/models.py          TextCNN and BiLSTM
src/train.py           train/evaluate one (input type, model) configuration
src/run_all.sh         all 6 RQ1/RQ2 configurations
src/rq3_transfer.py    RQ3 domain-transfer subsets
src/rq4_analysis.py    RQ4 feature analysis + non-neural baselines
src/make_figures.py    report figures
report/report.md       project report source
results/               metrics JSON, predictions, trained models
```

## Reproduce

```bash
python3 -m venv venv
venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install -r requirements.txt

venv/bin/python src/build_dataset.py    # downloads ~1.6 GB of parquet shards
venv/bin/python src/make_splits.py
./src/run_all.sh                        # 6 CNN/LSTM training runs (RQ1/RQ2)
venv/bin/python src/tfidf_baseline.py   # TF-IDF + logistic regression baseline
venv/bin/python src/rq3_transfer.py     # build cross-domain subsets
./src/run_extra.sh                      # RQ3 transfer runs + RQ4 ablation
venv/bin/python src/rq4_analysis.py     # feature analysis + baselines
venv/bin/python src/make_figures.py
./src/export_report.sh                  # report.docx / report.pdf
```

On CPU only; training all 6 main configurations plus extras takes roughly
3-5 hours. Everything uses seed 448. `src/run_extended.sh` is an
alternative driver that runs the same protocol with resumable logs.

## Results summary

See `report/report.pdf`. Response-only classification reaches **83.3%**
accuracy (CNN, 6 balanced classes, chance = 16.7%); prompt-only is at
chance, i.e. the prompt does not reveal which model will answer. The
BiLSTM reaches 70.0%. Cross-domain transfer, feature analyses, and a
TF-IDF baseline are in the report.
