from __future__ import annotations

from .attention import attention


def layer_norm(vector: list[float], eps: float = 1e-6) -> list[float]:
    mean = sum(vector) / len(vector)
    variance = sum((value - mean) ** 2 for value in vector) / len(vector)
    scale = (variance + eps) ** 0.5
    return [(value - mean) / scale for value in vector]


def feed_forward(vector: list[float]) -> list[float]:
    return [max(0.0, value) * 0.5 + value * 0.1 for value in vector]


def transformer_block(query: list[float], keys: list[list[float]], values: list[list[float]]) -> dict[str, list[float]]:
    attended, weights = attention(query, keys, values)
    residual = [q + a for q, a in zip(query, attended)]
    normalized = layer_norm(residual)
    ff = feed_forward(normalized)
    output = [n + f for n, f in zip(normalized, ff)]
    return {"attention_output": attended, "attention_weights": weights, "normalized": normalized, "output": output}
