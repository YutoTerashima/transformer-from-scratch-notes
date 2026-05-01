import json
from pathlib import Path


def test_real_imdb_stats_have_nontrivial_lengths():
    rows = [json.loads(line) for line in Path("datasets/external/imdb_token_stats.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 200
    assert max(row["token_count"] for row in rows) > 100
    assert len({row["label"] for row in rows}) >= 1
