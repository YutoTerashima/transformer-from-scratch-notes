from pathlib import Path

from transformer_from_scratch_notes.checkpoint import save_checkpoint


if __name__ == "__main__":
    save_checkpoint(Path("reports/demo_checkpoint.json"), 100, {"wq": [0.1, 0.2]}, {"loss": 0.42})
    print("reports/demo_checkpoint.json")
