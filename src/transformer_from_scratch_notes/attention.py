from __future__ import annotations

import math


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def attention(query: list[float], keys: list[list[float]], values: list[list[float]]) -> tuple[list[float], list[float]]:
    scale = math.sqrt(len(query))
    weights = softmax([dot(query, key) / scale for key in keys])
    output = [sum(weight * value[i] for weight, value in zip(weights, values)) for i in range(len(values[0]))]
    return output, weights
