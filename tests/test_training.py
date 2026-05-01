from transformer_from_scratch_notes.training import next_token_counts, predict_next


def test_next_token_prediction():
    counts = next_token_counts([1, 2, 1, 2, 1, 3])
    assert predict_next(1, counts) == 2
