from __future__ import annotations


def next_token_counts(tokens: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for left, right in zip(tokens, tokens[1:]):
        counts[(left, right)] = counts.get((left, right), 0) + 1
    return counts


def predict_next(current: int, counts: dict[tuple[int, int], int]) -> int | None:
    candidates = [(right, count) for (left, right), count in counts.items() if left == current]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]
