from transformer_from_scratch_notes.block import transformer_block


if __name__ == "__main__":
    result = transformer_block([1.0, 0.2], [[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 0.5]])
    print({key: [round(x, 3) for x in value] for key, value in result.items()})
