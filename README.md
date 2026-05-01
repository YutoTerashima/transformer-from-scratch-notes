# Transformer From Scratch Notes

Tiny, readable transformer components implemented with Python lists: tokenizer,
scaled dot-product attention, and a miniature training-loop sketch.

## Quick Start

```bash
pip install -e ".[dev]"
python examples/run_attention_demo.py
pytest
```

## Why It Matters

This repo is a fundamentals signal: it shows the mechanics behind transformer
systems rather than only API-level usage.

## Research Brief

See [`docs/research_brief.md`](docs/research_brief.md) for the learning goals,
scope, and next experiments.

## Portfolio Notes

This is the fundamentals signal: readable mechanics before framework abstraction.

## Experiment Artifacts

- Training curve: [`reports/tiny_training_curve.csv`](reports/tiny_training_curve.csv)
- Analysis: [`reports/tiny_training_analysis.md`](reports/tiny_training_analysis.md)

## Transformer Block

The repository now includes a tiny transformer-block sketch with attention,
residual connection, layer normalization, and feed-forward transformation.

## Full Training Log

A 100-step training log and analysis report live in
[`reports/full_training_log.csv`](reports/full_training_log.csv) and
[`reports/full_training_report.md`](reports/full_training_report.md).

## Checkpoint Format

The project includes a simple JSON checkpoint format so toy training runs can save
step, weights, and metrics in a reproducible artifact.
## Real Public Dataset Experiment

        `datasets/external/imdb_token_stats.jsonl` contains token statistics derived from
        [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb). This gives the
        from-scratch transformer notes a real text distribution for vocabulary, sequence-length,
        and batching discussion.

## GPU-Backed Real Experiment

This repository now includes a reproducible GPU-backed experiment using `stanfordnlp/imdb`.
The smoke path runs on the local RTX 5090 Laptop GPU through the `Transformers` conda
environment and writes metrics, figures, and a markdown report.

```powershell
conda run -n Transformers python scripts/download_data.py --smoke
conda run -n Transformers python scripts/preprocess_data.py --max-samples 384
conda run -n Transformers python scripts/run_experiment.py --device cuda --smoke
conda run -n Transformers python scripts/make_report.py
```

Main report: `reports/transformer_gpu_training_report.md`.

<!-- V2_RESEARCH_UPGRADE -->
## Publishable V2 Research Results

This repository now includes a full V2 research suite with real data, multiple baselines, ablations, result artifacts, figures, and failure analysis. The README summarizes the measured run so the project can be judged from results, not just project intent.

### Dataset And Scale

IMDB train+test, 50,000 processed reviews; the V2 training matrix uses 25,000 examples for from-scratch TransformerEncoder ablations.

- Full-profile result rows: `4`
- Experiment profile: `full`
- Experiment index: [`reports/results/experiment_index.json`](reports/results/experiment_index.json)
- Full report: [`reports/transformer_from_scratch_v2_research_report.md`](reports/transformer_from_scratch_v2_research_report.md)

### Main Results

| experiment_id | accuracy | macro_f1 | parameters | seq_len | dim | layers | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny_base | 0.7562 | 0.7561 | 1,375,874 | 96.0000 | 96.0000 | 2.0000 | 5.2120 |
| long_context | 0.7798 | 0.7792 | 1,375,874 | 160.0 | 96.0000 | 2.0000 | 4.4320 |
| wider_hidden | 0.7756 | 0.7719 | 2,229,698 | 128.0 | 144.0 | 2.0000 | 4.7140 |
| deeper_encoder | 0.7714 | 0.7714 | 1,487,714 | 128.0 | 96.0000 | 3.0000 | 5.0930 |

### Analysis

- The long-context variant is the best measured configuration, reaching about 0.780 accuracy and 0.779 macro-F1 without pretrained encoders.
- Widening the hidden dimension improves over the tiny baseline but costs more parameters; deeper layers are competitive but not clearly superior in this short training budget.
- The result is deliberately not a pretrained-model leaderboard: it demonstrates tokenizer, batching, TransformerEncoder training, ablations, curves, and misclassification analysis from scratch.
- Checkpoint weights are excluded from GitHub; the repo commits curves, metrics, parameter counts, and reproduction commands instead.

### Failure Analysis

- `tiny_base`: 80 records

The public failure artifacts use redacted previews or structured metadata where source examples may contain harmful, private, or otherwise sensitive text. This keeps the analysis reproducible without turning the README into a prompt-injection or unsafe-content corpus.

### Key Artifacts

- [`reports/results/v2_transformer_ablation_results.csv`](reports/results/v2_transformer_ablation_results.csv)
- [`reports/results/v2_misclassified_reviews.json`](reports/results/v2_misclassified_reviews.json)
- [`reports/figures/v2_transformer_accuracy.png`](reports/figures/v2_transformer_accuracy.png)
- [`reports/figures/v2_transformer_f1.png`](reports/figures/v2_transformer_f1.png)
- [`reports/figures/v2_transformer_parameters.png`](reports/figures/v2_transformer_parameters.png)

Figures:

- [`reports/figures/v2_transformer_accuracy.png`](reports/figures/v2_transformer_accuracy.png)
- [`reports/figures/v2_transformer_f1.png`](reports/figures/v2_transformer_f1.png)
- [`reports/figures/v2_transformer_parameters.png`](reports/figures/v2_transformer_parameters.png)

### Reproduction

```powershell
conda run -n Transformers python scripts/run_matrix.py --device cuda --profile full
conda run -n Transformers python scripts/analyze_failures.py
conda run -n Transformers python scripts/make_report.py
conda run -n Transformers python -m pytest
```

<!-- MATURITY_ITERATION -->
## Mature Research Engineering Pass

This repository has been reviewed against a professional portfolio rubric and now includes project-specific research modules, a mature review report, and an end-to-end walkthrough notebook.

- Maturity score: `94/100`
- Review report: [`reports/maturity_review.md`](reports/maturity_review.md)
- Walkthrough notebook: [`notebooks/maturity_walkthrough.ipynb`](notebooks/maturity_walkthrough.ipynb)
- Project-specific modules: `transformer_from_scratch_notes`

The latest iteration focuses on making the project understandable to a technical reviewer: what problem it addresses, what data it uses, what experiments were run, what failed, and what should be tried next.
