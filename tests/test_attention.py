from transformer_from_scratch_notes.attention import attention
from transformer_from_scratch_notes.tokenizer import TinyTokenizer


def test_attention_weights_sum_to_one():
    _, weights = attention([1, 0], [[1, 0], [0, 1]], [[1, 0], [0, 1]])
    assert round(sum(weights), 6) == 1.0


def test_tokenizer_is_stable():
    tokenizer = TinyTokenizer()
    assert tokenizer.encode("AI AI") == [0, 0]
