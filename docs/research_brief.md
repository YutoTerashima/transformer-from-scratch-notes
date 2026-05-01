# Research Brief

## Problem

Transformer projects often hide the core mechanism behind frameworks. This repository
keeps the mechanics visible: tokenization, attention weights, and tiny next-token
statistics.

## Method

The code intentionally uses small Python data structures so each computation can be
read directly.

## Depth Signal

The project connects attention math to next-token prediction intuition without pretending
to be a production training stack.

## Next Experiments

- Add positional encoding notes.
- Add a NumPy implementation.
- Add a tiny character-level training loop.
