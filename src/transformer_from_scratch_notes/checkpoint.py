from __future__ import annotations

import json
from pathlib import Path


def save_checkpoint(path: Path, step: int, weights: dict[str, list[float]], metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"step": step, "weights": weights, "metrics": metrics}, indent=2), encoding="utf-8")


def load_checkpoint(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
