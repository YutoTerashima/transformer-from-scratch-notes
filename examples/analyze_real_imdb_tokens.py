import json
from pathlib import Path

rows = [json.loads(line) for line in Path("datasets/external/imdb_token_stats.jsonl").read_text(encoding="utf-8").splitlines()]
print({"rows": len(rows), "avg_tokens": round(sum(r["token_count"] for r in rows) / len(rows), 2), "max_tokens": max(r["token_count"] for r in rows)})
