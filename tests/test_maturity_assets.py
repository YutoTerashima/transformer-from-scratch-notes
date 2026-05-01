from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_project_specific_maturity_modules_import():
    import transformer_from_scratch_notes.tokenizer_ablation as _tokenizer_ablation
    import transformer_from_scratch_notes.attention_export as _attention_export
    import transformer_from_scratch_notes.training_diagnostics as _training_diagnostics


def test_mature_review_assets_exist():
    report = REPO / "reports" / "maturity_review.md"
    notebook = REPO / "notebooks" / "maturity_walkthrough.ipynb"
    assert report.exists()
    assert notebook.exists()
    text = report.read_text(encoding="utf-8")
    for heading in ["Abstract", "Dataset Card", "Methods", "Results", "Failure Analysis", "Maturity Review", "Reproduction"]:
        assert f"## {heading}" in text
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 4
