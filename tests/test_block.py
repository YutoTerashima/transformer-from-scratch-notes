from transformer_from_scratch_notes.block import transformer_block


def test_transformer_block_outputs_vector():
    result = transformer_block([1, 0], [[1, 0], [0, 1]], [[1, 0], [0, 1]])
    assert len(result["output"]) == 2
    assert round(sum(result["attention_weights"]), 6) == 1.0
