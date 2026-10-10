from preparation.dolly import dolly_messages


def test_a_dolly_row_becomes_one_exchange_of_context_then_instruction():
    row = {
        "instruction": "Who founded the company?",
        "context": "Acme was founded in 1901 by Jane Roe.",
        "response": "Jane Roe.",
        "category": "closed_qa",
    }

    assert dolly_messages(row) == {
        "messages": [
            {"role": "user", "content": row["context"] + "\n\n" + row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ]
    }
