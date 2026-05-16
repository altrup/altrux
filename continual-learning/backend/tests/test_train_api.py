def test_train_critic_returns_loss_and_step(client):
    resp = client.post("/train/critic", json={
        "prompt": "hello",
        "response": " world",
        "user_reward": 0.5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "loss" in data
    assert data["step"] == 1


def test_train_policy_returns_reward_loss_step(client):
    resp = client.post("/train/policy", json={
        "prompt": "hello",
        "response": " world",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "reward" in data
    assert "loss" in data
    assert data["step"] == 1


def test_train_history_empty_initially(client):
    resp = client.get("/train/history")
    assert resp.status_code == 200
    assert resp.json()["entries"] == []


def test_train_history_grows(client):
    client.post("/train/critic", json={"prompt": "a", "response": "b", "user_reward": 0.0})
    client.post("/train/policy", json={"prompt": "a", "response": "b"})
    resp = client.get("/train/history", params={"n": 10})
    entries = resp.json()["entries"]
    assert len(entries) == 2
    assert entries[0]["phase"] == 1
    assert entries[1]["phase"] == 2


def test_train_history_n_limits_results(client):
    for _ in range(5):
        client.post("/train/critic", json={"prompt": "a", "response": "b", "user_reward": 0.1})
    resp = client.get("/train/history", params={"n": 3})
    assert len(resp.json()["entries"]) == 3


def test_checkpoint_status(client):
    resp = client.get("/checkpoint/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "step" in data
    assert "checkpoint_dir" in data


def test_critic_user_reward_validation(client):
    resp = client.post("/train/critic", json={
        "prompt": "a",
        "response": "b",
        "user_reward": 2.0,
    })
    assert resp.status_code == 422
