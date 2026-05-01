# Tiny Training Analysis

This is a deterministic toy training curve used to document how the repository
would report learning dynamics. It is intentionally small, but it has the same
artifact shape as a real training note: step, loss, and attention entropy.

## First and Last Measurements

| step | loss | attention_entropy |
| --- | --- | --- |
| 1 | 2.1168 | 1.788 |
| 40 | 0.4977 | 1.32 |

## Interpretation

Loss decreases steadily while attention entropy narrows. In a real experiment,
this would suggest the model is moving from diffuse attention toward more specific
token associations.
