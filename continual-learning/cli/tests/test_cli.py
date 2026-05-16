"""CLI tests using Click's test runner and mocked httpx responses."""
import json
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from cl_cli.main import cli


@pytest.fixture()
def runner():
    return CliRunner()


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = data
    m.raise_for_status = MagicMock()
    return m


def test_status_command(runner):
    mock_resp = _mock_response({"step": 5, "checkpoint_dir": "../checkpoints", "last_saved_step": None})
    with patch("httpx.get", return_value=mock_resp):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "Step:" in result.output
    assert "5" in result.output


def test_save_command(runner):
    mock_resp = _mock_response({"ok": True, "step": 10})
    with patch("httpx.post", return_value=mock_resp):
        result = runner.invoke(cli, ["save"])
    assert result.exit_code == 0
    assert "10" in result.output


def test_train_critic_command(runner):
    mock_resp = _mock_response({"loss": 0.25, "step": 1})
    with patch("httpx.post", return_value=mock_resp):
        result = runner.invoke(cli, [
            "train", "critic",
            "--prompt", "hello",
            "--response", " world",
            "--reward", "0.5",
        ])
    assert result.exit_code == 0
    assert "loss" in result.output.lower()


def test_train_policy_command(runner):
    mock_resp = _mock_response({"reward": 0.6, "loss": -0.6, "step": 2})
    with patch("httpx.post", return_value=mock_resp):
        result = runner.invoke(cli, [
            "train", "policy",
            "--prompt", "hello",
            "--response", " world",
        ])
    assert result.exit_code == 0
    assert "reward" in result.output.lower()


def test_history_empty(runner):
    mock_resp = _mock_response({"entries": []})
    with patch("httpx.get", return_value=mock_resp):
        result = runner.invoke(cli, ["history"])
    assert result.exit_code == 0
    assert "No training history" in result.output


def test_history_with_entries(runner):
    entries = [
        {"step": 1, "phase": 1, "user_reward": 0.5, "predicted_reward": 0.3, "loss": 0.04},
        {"step": 2, "phase": 2, "user_reward": None, "predicted_reward": 0.4, "loss": -0.4},
    ]
    mock_resp = _mock_response({"entries": entries})
    with patch("httpx.get", return_value=mock_resp):
        result = runner.invoke(cli, ["history", "--n", "5"])
    assert result.exit_code == 0
    assert "Phase" in result.output


def test_generate_no_stream(runner):
    mock_resp = _mock_response({"response": "Hello there!", "estimated_reward": 0.75})
    with patch("httpx.post", return_value=mock_resp):
        result = runner.invoke(cli, ["generate", "--no-stream", "Hello"])
    assert result.exit_code == 0
    assert "Hello there!" in result.output


def test_reward_out_of_range_rejected(runner):
    result = runner.invoke(cli, [
        "train", "critic",
        "--prompt", "a",
        "--response", "b",
        "--reward", "2.0",
    ])
    assert result.exit_code != 0


def test_server_option(runner):
    mock_resp = _mock_response({"step": 0, "checkpoint_dir": ".", "last_saved_step": None})
    with patch("httpx.get", return_value=mock_resp) as mock_get:
        runner.invoke(cli, ["--server", "http://myserver:9000", "status"])
    call_url = mock_get.call_args[0][0]
    assert "myserver:9000" in call_url
