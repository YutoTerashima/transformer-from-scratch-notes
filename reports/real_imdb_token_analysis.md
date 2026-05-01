# Real Corpus Analysis: IMDB Reviews

Source: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)

This experiment profiles 240 real IMDB training reviews for tokenizer and
tiny-language-model experiments.

- Average tokens per review: 228.812
- Median tokens per review: 167.5
- Most frequent tokens: [('the', 3061), ('a', 1555), ('and', 1439), ('of', 1365), ('to', 1347), ('br', 1024), ('is', 937), ('in', 923), ('this', 808), ('i', 783)]

Interpretation: even a tiny transformer demo benefits from real corpus statistics because
sequence length, vocabulary skew, and label balance affect batching and loss curves.
