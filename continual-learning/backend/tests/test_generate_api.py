import json


def test_generate_returns_response_and_reward(client):
    resp = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 16})
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "estimated_reward" in data
    assert -1.0 <= data["estimated_reward"] <= 1.0


def test_generate_stream_produces_sse_events(client):
    with client.stream("GET", "/generate/stream", params={"prompt": "hi", "max_new_tokens": 16}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = []
        for line in resp.iter_lines():
            if line:
                lines.append(line)
            if line == "data: [DONE]":
                break

    assert any(line.startswith("data:") for line in lines)
    done_lines = [l for l in lines if l == "data: [DONE]"]
    assert done_lines, "SSE stream did not send [DONE]"

    payload_lines = [l for l in lines if l.startswith("data:") and l != "data: [DONE]"]
    parsed_events = [json.loads(l[len("data: "):]) for l in payload_lines]

    # First event must be the prefill ack with token list
    assert parsed_events[0].get("status") == "prefill"
    assert "tokens" in parsed_events[0]

    # Remaining events are either prefill-layer progress or chunk events
    chunk_events = [e for e in parsed_events if "chunk" in e]
    assert chunk_events, "No chunk events in SSE stream"


def test_generate_stream_final_event_has_reward(client):
    chunks = []
    with client.stream("GET", "/generate/stream", params={"prompt": "hi", "max_new_tokens": 4}) as resp:
        for line in resp.iter_lines():
            if line.startswith("data:") and line != "data: [DONE]":
                chunks.append(json.loads(line[len("data: "):]))
            if line == "data: [DONE]":
                break

    reward_events = [c for c in chunks if "estimated_reward" in c]
    assert reward_events, "No final reward event in SSE stream"
    assert -1.0 <= reward_events[-1]["estimated_reward"] <= 1.0
