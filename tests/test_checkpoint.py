from pathlib import Path

from transformer_from_scratch_notes.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "ckpt.json"
    save_checkpoint(path, 1, {"w": [1.0]}, {"loss": 0.5})
    assert load_checkpoint(path)["metrics"]["loss"] == 0.5
