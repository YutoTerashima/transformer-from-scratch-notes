# Transformer From Scratch V2 Research Report

## Abstract

This V2 upgrade turns the repository into a reproducible project-level experiment suite. The run records the dataset, device, experiment matrix, metrics, figures, failure analysis, and reproduction commands in committed small artifacts.

## Dataset

- Source path: `data/processed/imdb_examples.jsonl`
- Profile: `full`
- Runtime: `20.535` seconds
- Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`

## Methods

Experiments declared in `configs/experiment_matrix.yaml`:

- `tiny_base`: `{'seq_len': 96, 'dim': 96, 'layers': 2, 'heads': 4, 'dropout': 0.1}`
- `long_context`: `{'seq_len': 160, 'dim': 96, 'layers': 2, 'heads': 4, 'dropout': 0.1}`
- `wider_hidden`: `{'seq_len': 128, 'dim': 144, 'layers': 2, 'heads': 4, 'dropout': 0.1}`
- `deeper_encoder`: `{'seq_len': 128, 'dim': 96, 'layers': 3, 'heads': 4, 'dropout': 0.15}`

## Experiments

The matrix produced `4` result rows. Best observed `macro_f1`: `0.7792` from `long_context`.

## Results

Key artifacts:

- `reports\results\v2_transformer_ablation_results.csv`
- `reports\results\v2_misclassified_reviews.json`
- `reports\figures\v2_transformer_accuracy.png`
- `reports\figures\v2_transformer_f1.png`
- `reports\figures\v2_transformer_parameters.png`

## Ablations

Configured ablations: sequence_length, hidden_size, layers, dropout. The generated ablation files quantify threshold, perturbation, architecture, retrieval, or metric sensitivity depending on the project.

## Failure Analysis

Failure records: `80`.

Top clusters:

- `tiny_base`: 80

## Discussion

This project keeps the model from scratch. V2 measures architecture tradeoffs on IMDB without using pretrained encoders, so the results reflect model design rather than transferred representations.

## Limitations

- Full raw caches, model weights, and optimizer states are intentionally excluded from GitHub.
- Results are designed for reproducible portfolio research; they are not production safety, medical, or compliance guarantees.
- Some V2 experiments use compact local artifacts to keep the repository lightweight.

## Reproduction

```powershell
conda run -n Transformers python scripts/run_matrix.py --device cuda --profile full
conda run -n Transformers python scripts/analyze_failures.py
conda run -n Transformers python scripts/make_report.py
conda run -n Transformers python -m pytest
```
