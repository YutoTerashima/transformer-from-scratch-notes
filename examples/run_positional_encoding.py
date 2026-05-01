from transformer_from_scratch_notes.positional import sinusoidal_position


if __name__ == "__main__":
    print([round(x, 4) for x in sinusoidal_position(3, 6)])
