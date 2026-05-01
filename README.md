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
## Publishable V2 Research Upgrade

This repository now includes a project-level V2 experiment suite:

- Reproducible matrix: `configs/experiment_matrix.yaml`
- Main runner: `scripts/run_matrix.py --device cuda --profile full`
- Failure analysis: `scripts/analyze_failures.py`
- Research report: `reports/transformer_from_scratch_v2_research_report.md`
- Experiment index: `reports/results/experiment_index.json`

The V2 artifacts include multiple experiments, ablations, figures, failure cases, and a discussion section while keeping raw caches and large checkpoints out of Git.

