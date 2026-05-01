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
