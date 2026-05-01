from transformer_from_scratch_notes.tokenizer import TinyTokenizer
from transformer_from_scratch_notes.training import next_token_counts, predict_next


if __name__ == "__main__":
    tokenizer = TinyTokenizer()
    tokens = tokenizer.encode("ai agents evaluate ai agents")
    counts = next_token_counts(tokens)
    print("vocab", tokenizer.vocab)
    print("predict_after_ai", predict_next(tokenizer.vocab["ai"], counts))
