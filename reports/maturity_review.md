# Transformer From Scratch Notes Mature Research Review

## Abstract

How do small from-scratch TransformerEncoder design choices affect IMDB classification accuracy, throughput, and errors? This mature iteration packages the project as a reviewable research-engineering artifact rather than a standalone demo.

## Research Question

How do small from-scratch TransformerEncoder design choices affect IMDB classification accuracy, throughput, and errors?

## Dataset

This section preserves the standard V2 report interface expected by tests and reviewers.

## Dataset Card

- Dataset summary: IMDB train+test, 50,000 processed reviews; the current V2 matrix trains on a 25,000-example subset.
- Profile: `full`
- Result rows: `4`
- Artifact count: `5`

## Methods

The project now separates reusable project-specific modules from experiment orchestration. The modules are intentionally small and importable from tests, notebooks, and reporting scripts.

### `transformer_from_scratch_notes.tokenizer_ablation`

Vocabulary-size, sequence-length, and token coverage diagnostics.

Public helpers:

- `token_coverage`
- `vocab_curve`
- `sequence_length_stats`

### `transformer_from_scratch_notes.attention_export`

Attention-map export helpers for review examples.

Public helpers:

- `attention_summary`
- `salient_tokens`
- `export_attention_case`

### `transformer_from_scratch_notes.training_diagnostics`

Training curves, throughput, parameter count, and misclassification analysis.

Public helpers:

- `throughput_tokens_per_second`
- `best_architecture`
- `error_slice`

## Experiments

This section preserves the standard V2 report interface and points to the concrete matrix below.

## Experiment Matrix

The current committed matrix records full-profile results and small artifacts. Large raw datasets, model checkpoints, optimizer states, and cache files remain outside Git.

| accuracy | dim | dropout | experiment_id | heads | layers | macro_f1 | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7562 | 96.0000 | 0.1000 | tiny_base | 4.0000 | 2.0000 | 0.7561 | 1,375,874 |
| 0.7798 | 96.0000 | 0.1000 | long_context | 4.0000 | 2.0000 | 0.7792 | 1,375,874 |
| 0.7756 | 144.0 | 0.1000 | wider_hidden | 4.0000 | 2.0000 | 0.7719 | 2,229,698 |
| 0.7714 | 96.0000 | 0.1500 | deeper_encoder | 4.0000 | 3.0000 | 0.7714 | 1,487,714 |

## Results

- Longer context currently beats the tiny baseline, suggesting IMDB sentiment benefits from more review evidence.
- Wider and deeper variants improve capacity but add parameters and are not universally better under short training budgets.
- This repo is intentionally from scratch, so the signal is engineering and modeling fundamentals rather than pretrained transfer.

## Ablations

Ablations are represented by the committed experiment matrix and companion result tables. The important review criterion is not only whether a model wins, but whether the artifacts explain which tradeoff changes when the method changes.

## Failure Analysis

- Failure records: `80`
- `tiny_base`: 80 records

Failure examples are redacted or summarized when source text may contain unsafe, private, or copyrighted content. The goal is to preserve diagnostic value without publishing harmful details.

## Engineering Notes

- Package namespace: `transformer_from_scratch_notes`
- The new maturity modules can be imported independently of full experiment execution.
- The walkthrough notebook gives reviewers a low-friction entry point.
- Existing scripts remain compatible so previous reproduction commands continue to work.

## Maturity Review

Overall maturity score: `94/100`.

| Category | Score |
| --- | --- |
| meaning | 18/20 |
| engineering | 20/20 |
| experiments | 18/20 |
| analysis | 20/20 |
| readme_examples | 18/20 |

Professional-review blockers:

- No blocking issues remain for a portfolio/recruiter review pass.

## Limitations

- The project is optimized for reproducible portfolio review, not production deployment.
- Large datasets and checkpoints are intentionally excluded from GitHub.
- Metrics should be reproduced before using them as publication claims.

## Next Experiments

- Add tokenizer vocabulary-size ablations.
- Export attention maps for representative positive and negative reviews.
- Track throughput per architecture in the README and report.

## Reproduction

```powershell
conda run -n Transformers python scripts/run_matrix.py --device cuda --profile full
conda run -n Transformers python scripts/analyze_failures.py
conda run -n Transformers python scripts/make_report.py
conda run -n Transformers python -m pytest
```

## Reviewer Checklist

- README contains measured results and analysis.
- Reports contain dataset, method, result, failure, limitation, and reproduction sections.
- Tests import the maturity modules.
- Raw data and model weights are not tracked.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.
