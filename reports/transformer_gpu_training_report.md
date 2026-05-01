# Transformer GPU Training Report

From-scratch TransformerEncoder classifier trained on real IMDB text.

## Dataset

- Source: `stanfordnlp/imdb`
- Config: `plain_text`
- Split: `train`

## Reproducibility

```powershell
conda run -n Transformers python scripts/download_data.py --smoke
conda run -n Transformers python scripts/preprocess_data.py --max-samples 384
conda run -n Transformers python scripts/run_experiment.py --device cuda --smoke
conda run -n Transformers python scripts/make_report.py
```

## Generated Artifacts

- Result JSON: `results/tiny_transformer_metrics.json`
- Result CSV: `results/tiny_transformer_curve.csv`
- Figure: `figures/tiny_transformer_curve.png`

## Result Snapshot

```json
{
  "dataset": "stanfordnlp/imdb",
  "rows": 384,
  "vocab_size": 6145,
  "parameters": 739682,
  "device": {
    "requested_device": "cuda",
    "actual_device": "cuda",
    "cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "torch_version": "2.10.0+cu128",
    "cuda_runtime": "12.8"
  },
  "seconds": 0.479,
  "history": [
    {
      "epoch": 1,
      "loss": 0.0846,
      "val_accuracy": 1.0
    },
    {
      "epoch": 2,
      "loss": 0.0003,
      "val_accuracy": 1.0
    }
  ]
}
```

## Failure Analysis

The experiment stores model disagreements, retrieval misses, or policy-risk examples in the result JSON/CSV files when available. These examples are intentionally kept as previews or structured metadata where the source data can contain unsafe or sensitive text.

## Limitations

- Smoke mode prioritizes reproducibility and runtime over leaderboard-scale performance.
- Raw datasets are downloaded to `data/raw/` and are not committed.
- Metrics should be interpreted as portfolio research baselines, not production claims.
