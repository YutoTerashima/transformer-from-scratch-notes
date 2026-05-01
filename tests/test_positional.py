from transformer_from_scratch_notes.positional import sinusoidal_position


def test_positional_encoding_shape():
    assert len(sinusoidal_position(0, 8)) == 8
