from __future__ import annotations

import math


def sinusoidal_position(position: int, dimensions: int) -> list[float]:
    values: list[float] = []
    for i in range(dimensions):
        angle = position / (10000 ** (2 * (i // 2) / dimensions))
        values.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
    return values
