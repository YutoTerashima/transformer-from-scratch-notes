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
