import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2]))

from training import checkpoints
import train


def test_train_checkpoint_wrappers_pass_its_checkpoint_directory(monkeypatch, tmp_path):
    seen: list[Path] = []
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    monkeypatch.setattr(checkpoints, "latest_checkpoint", lambda ckpt_dir: seen.append(ckpt_dir))

    assert train.latest_checkpoint() is None
    assert seen == [tmp_path]


def test_train_reexports_checkpoint_loader():
    assert train.load_checkpoint is checkpoints.load_checkpoint
