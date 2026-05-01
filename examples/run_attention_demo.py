from transformer_from_scratch_notes.attention import attention


if __name__ == "__main__":
    output, weights = attention([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [[10.0, 0.0], [0.0, 5.0]])
    print("weights=", [round(item, 3) for item in weights])
    print("output=", [round(item, 3) for item in output])
