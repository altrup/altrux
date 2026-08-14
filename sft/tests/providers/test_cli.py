import json
import os
import subprocess
import sys

import pytest


def _fake_cli(tmp_path):
    path = tmp_path / "fake-cli"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "a=sys.argv[1:]\n"
        "if a == ['--version']: print('fake 1.2.3'); raise SystemExit\n"
        "prompt=sys.stdin.read()\n"
        "if '--output-format' in a:\n"
        " assert a[a.index('--tools')+1] == '' and '--strict-mcp-config' in a\n"
        " s='claude-session'\n"
        " if '--resume' in a: s=a[a.index('--resume')+1]\n"
        " print(json.dumps({'result':'user from claude','session_id':s,'usage':{'input_tokens':4,'output_tokens':3}}))\n"
        "elif 'exec' in a:\n"
        " assert 'approval_policy=\"never\"' in a\n"
        " assert '--skip-git-repo-check' in a\n"
        " s='codex-thread'\n"
        " if 'resume' in a: s=a[-2]\n"
        " else: assert a[a.index('--sandbox')+1] == 'read-only'\n"
        " print(json.dumps({'type':'thread.started','thread_id':s}))\n"
        " print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'user from codex'}}))\n"
        " print(json.dumps({'type':'turn.completed','usage':{'input_tokens':5,'output_tokens':2}}))\n"
        "else: raise SystemExit(9)\n"
    )
    os.chmod(path, 0o755)
    return path


@pytest.mark.parametrize("provider,message,session", [
    ("claude", "user from claude", "claude-session"),
    ("codex", "user from codex", "codex-thread"),
])
def test_cli_adapter_starts_and_resumes_native_sessions(tmp_path, provider, message, session):
    fake = _fake_cli(tmp_path)
    work_dir = tmp_path / "stable-work"
    command = [sys.executable, "-m", "providers.user_generator", "--provider", provider,
               "--model", "fake-model", "--executable", str(fake), "--work-dir", str(work_dir)]

    started = subprocess.run(command, input=json.dumps({"scenario": "s", "goal": "g", "turn": 1,
                                                        "latest_assistant_reply": ""}),
                             text=True, capture_output=True, check=True)
    first = json.loads(started.stdout)
    resumed = subprocess.run(command, input=json.dumps({"scenario": "s", "goal": "g2", "turn": 2,
                                                        "latest_assistant_reply": "reply",
                                                        "session_id": first["session_id"]}),
                             text=True, capture_output=True, check=True)
    second = json.loads(resumed.stdout)

    assert (first["message"], first["session_id"], first["resume_status"]) == (message, session, "started")
    assert (second["session_id"], second["resume_status"]) == (session, "resumed")
    assert second["provider"] == provider and second["model"] == "fake-model"
    assert second["version"] == "fake 1.2.3" and second["token_usage"]
    assert first["work_dir"] == second["work_dir"] == str(work_dir)


def test_cli_adapter_propagates_native_failure(tmp_path):
    path = tmp_path / "bad-cli"
    path.write_text("#!/bin/sh\n[ \"$1\" = --version ] && { echo bad-1; exit 0; }; exit 7\n")
    os.chmod(path, 0o755)

    completed = subprocess.run(
        [sys.executable, "-m", "providers.user_generator", "--provider", "claude", "--model", "m",
         "--executable", str(path)],
        input=json.dumps({"scenario": "s", "goal": "g", "turn": 1, "latest_assistant_reply": ""}),
        text=True, capture_output=True, check=False,
    )

    assert completed.returncode != 0
