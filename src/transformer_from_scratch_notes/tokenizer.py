class TinyTokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in text.lower().split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        return ids
