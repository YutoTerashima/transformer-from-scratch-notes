from __future__ import annotations

"""Report metadata for the mature portfolio iteration."""

PROJECT_TITLE = 'Transformer From Scratch Notes'
RESEARCH_PROBLEM = 'How do small from-scratch TransformerEncoder design choices affect IMDB classification accuracy, throughput, and errors?'
DATASET_SUMMARY = 'IMDB train+test, 50,000 processed reviews; the current V2 matrix trains on a 25,000-example subset.'
TAKEAWAYS = ['Longer context currently beats the tiny baseline, suggesting IMDB sentiment benefits from more review evidence.', 'Wider and deeper variants improve capacity but add parameters and are not universally better under short training budgets.', 'This repo is intentionally from scratch, so the signal is engineering and modeling fundamentals rather than pretrained transfer.']
NEXT_EXPERIMENTS = ['Add tokenizer vocabulary-size ablations.', 'Export attention maps for representative positive and negative reviews.', 'Track throughput per architecture in the README and report.']


def report_outline() -> list[str]:
    return [
        "Abstract",
        "Research question",
        "Dataset card",
        "Methods",
        "Experiment matrix",
        "Results",
        "Ablations",
        "Failure analysis",
        "Engineering notes",
        "Limitations",
        "Reproduction",
    ]


def maturity_claims() -> dict[str, object]:
    return {
        "title": PROJECT_TITLE,
        "problem": RESEARCH_PROBLEM,
        "dataset": DATASET_SUMMARY,
        "takeaways": TAKEAWAYS,
        "next_experiments": NEXT_EXPERIMENTS,
    }
