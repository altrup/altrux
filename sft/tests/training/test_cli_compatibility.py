import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2]))

from training import cli
import train


def test_train_main_passes_its_mutable_checkpoint_directory(monkeypatch, tmp_path):
    seen: list[tuple[Path, str]] = []
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    monkeypatch.setattr(cli, "main", lambda ckpt_dir, model_name: seen.append((ckpt_dir, model_name)))

    train.main()

    assert seen == [(tmp_path, train.MODEL_NAME)]


def test_train_reexports_cli_loaded_model_configuration():
    assert train.MODEL_ID == cli.load_model_runtime(train.MODEL_NAME).model_id
    assert train.hooks is cli.load_model_runtime(train.MODEL_NAME).hooks
