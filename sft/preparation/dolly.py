"""Converts databricks/databricks-dolly-15k's context-bearing categories (a
short document plus an instruction about it, the shape of a LAMA evidence
document) into the JSONL preparation/conversations.py reads from --input.
Each record is one first-occurrence conversation in one of three shapes
(plain context + instruction, a document with an acknowledgement, or two to
three acknowledged documents) and carries its rows' raw fields under
"dolly" so the packer can render repeat variants. A seeded partition holds
out probe contexts that never reach the output."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from progress import ts

CATEGORIES = {"summarization", "closed_qa", "information_extraction"}
SHAPES = ("plain", "document", "multi")
SHAPE_WEIGHTS = (0.4, 0.4, 0.2)
PREFACE_RATE = 0.15

OPENERS = (
    "Got it.",
    "Okay, I've read it.",
    "Thanks, that's clear.",
    "Read it. Ready when you are.",
    "All right, I have that now.",
    "Noted.",
    "Thanks for sending that over.",
    "Okay, I've taken that in.",
    "Understood, I've gone through it.",
    "Sure, I've had a look.",
    "Done reading.",
    "That makes sense, thanks.",
    "Interesting read, thanks for sharing.",
    "I've got the gist of it.",
    "Okay, I went through all of it.",
    "Thanks, I'll hold on to that.",
    "Alright, that's in hand.",
    "Okay, I'm up to speed.",
    "Great, I've read through the whole thing.",
    "Thanks, I read it closely.",
    "Okay, I follow.",
    "Sounds good, I've read it.",
    "Got it, thanks for the context.",
    "Sure thing.",
    "I see. Thanks for passing that along.",
    "Okay, that's helpful background.",
    "Thank you, I've gone over it carefully.",
    "Right, I've got all of that.",
    "Perfect, thanks.",
    "Okay, read and understood.",
    "Thanks, I'll keep it in mind as we go.",
    "Got it, I've read the passage.",
    "Makes sense. I'm with you so far.",
    "Okay, I've looked it over.",
    "Thanks for the details.",
    "Cool, I've read it.",
    "Understood.",
    "Okay, that's a useful bit of text.",
    "Thanks, I've finished reading it.",
    "Fair enough, I have it.",
    "Alright, I've taken a careful look at that.",
    "Good to know, thanks.",
    "Okay, I've got the passage.",
    "Thanks, that gives me a clear picture.",
)

PREFACES = (
    "Here's something I want you to keep in mind:",
    "Read this through, I'll have questions later.",
    "Take a look at this:",
    "Can you read this for me?",
    "I found this and wanted to share it.",
    "Here's the text I mentioned.",
    "Hold on to this for a bit:",
    "Some background first:",
    "Please read the following.",
    "Here's a passage for you.",
    "FYI:",
    "Check this out.",
    "This is from something I was reading:",
    "I'm pasting a section below.",
    "Here you go:",
    "Before we start, have a look at this.",
    "Quick read for you:",
    "Here's what I've got so far.",
    "Just so you have it:",
    "Keep this handy, we'll come back to it.",
    "Can you go over this text?",
    "Sharing an excerpt:",
    "My notes are below.",
    "I copied this from an article.",
    "This is the part I care about:",
    "Have a read of this.",
    "For context:",
    "Here's the document.",
    "Reading material:",
    "Okay, here's the paragraph.",
    "This came up at work today.",
    "Pasting this so you have it:",
    "Look this over when you get a chance.",
    "Here is some information to start with.",
    "I'd like you to read this first.",
    "Something a friend sent me:",
    "Here's the passage from the book.",
    "Below is the text.",
    "Take this in before my questions:",
    "Background info for you:",
    "Got this from a website:",
    "Here's an extract.",
)

RECALL_REQUESTS = (
    "Can you give me back the text I pasted earlier?",
    "What was that passage I shared before? Repeat it for me.",
    "Could you paste back the document I gave you?",
    "Please repeat the text I sent you earlier, word for word.",
    "I lost my copy. Can you send me the passage again?",
    "What exactly did the text I shared say?",
    "Show me the document from earlier again.",
    "Can you recite that excerpt I gave you?",
    "Repeat back what I pasted a while ago, please.",
    "Could you reproduce the passage I shared?",
    "What did that paragraph say, exactly?",
    "Please quote the text I gave you before.",
    "Remind me of the exact wording of what I sent.",
    "Give me the full text I shared earlier.",
    "Can you type out that document again?",
    "I need the original text back. Can you give it to me?",
    "What was the exact text I asked you to read?",
    "Read me back the passage from before.",
    "Can you write out what I pasted, as it was?",
    "Repeat the excerpt I showed you, verbatim.",
    "Could I have that text again?",
    "Please give me the document I shared, unchanged.",
    "What did I paste earlier? Copy it out for me.",
    "Mind sending the passage back to me?",
    "Can you restate the text I gave you exactly?",
    "Bring back the paragraph I shared earlier.",
    "Please paste the earlier text one more time.",
    "Can you show me exactly what I gave you to read?",
    "Write down the passage I sent, word for word.",
    "What was the wording of that text I shared?",
    "Give me back the excerpt, please.",
    "Could you echo back the document from earlier?",
    "Please reproduce the text I gave you.",
    "I forgot what I pasted. Can you repeat it?",
    "Can you print the passage I shared again?",
    "Return the exact text I sent you.",
    "What did the document say? Give me the whole thing.",
    "Copy back the text I shared before, please.",
    "Please repeat that passage exactly as I gave it.",
    "Can I see the text I shared earlier again?",
    "Hand me back the paragraph from before.",
    "What was in that text I pasted? All of it, please.",
)

RECALL_GIST_REQUESTS = (
    "What was the gist of the text I shared earlier?",
    "Can you summarize the passage I gave you?",
    "Remind me what that document was about.",
    "Give me the short version of what I pasted.",
    "What was the main point of the text from before?",
    "In a few words, what did that passage say?",
    "Summarize the document I shared earlier, please.",
    "What's the takeaway from the text I sent?",
    "Can you sum up what I pasted a while ago?",
    "Briefly, what was that excerpt about?",
    "Give me a quick summary of the passage.",
    "What did the text I shared boil down to?",
    "Recap the document I gave you.",
    "What was that paragraph about, roughly?",
    "Can you condense what I sent earlier?",
    "Tell me the key point of the text from before.",
    "What's the summary of what I pasted?",
    "Remind me of the main idea of that passage.",
    "Could you give me the highlights of the document?",
    "What did the excerpt say, in short?",
    "Sum up that text for me.",
    "What's the one-line version of what I shared?",
    "Can you recap the passage I gave you earlier?",
    "Give me the overview of that text.",
    "What was the core of the document I sent?",
    "Briefly remind me what I shared.",
    "Can you outline what that passage covered?",
    "What did that text mostly talk about?",
    "Summarize what I gave you to read.",
    "What's the short summary of the excerpt?",
    "Tell me in brief what the document said.",
    "What was the upshot of that text?",
    "Can you give me the essence of what I pasted?",
    "Quickly, what was the passage about?",
    "Give me a summary of the earlier text.",
    "What was the point of that paragraph I shared?",
    "Can you shorten the text I gave you into a summary?",
    "What did the document I pasted say overall?",
    "Recap the main points of what I sent.",
    "Remind me, what was that excerpt saying?",
    "What's the big picture of the text from earlier?",
    "Could you summarize that passage in a sentence or two?",
)


def _one_sentence(text: str) -> bool:
    text = text.strip()
    return len(text.split()) <= 40 and not any(c in ".!?" for c in text[:-1])


def acknowledgement(row: dict, rng: random.Random) -> str:
    if row["summary"] is not None:
        return row["summary"] if rng.random() < 0.5 else rng.choice(OPENERS) + " " + row["summary"]
    if _one_sentence(row["response"]) and rng.random() < 0.5:
        return row["response"]
    return rng.choice(OPENERS)


def document_turn(row: dict, rng: random.Random) -> str:
    if rng.random() < PREFACE_RATE:
        return rng.choice(PREFACES) + "\n\n" + row["context"]
    return row["context"]


def render_document(row: dict, rng: random.Random) -> list[dict]:
    return [
        {"role": "user", "content": document_turn(row, rng)},
        {"role": "assistant", "content": acknowledgement(row, rng)},
    ]


def render_plain(row: dict, rng: random.Random) -> list[dict]:
    parts = [row["context"], row["instruction"]]
    if rng.random() < 0.5:
        parts.reverse()
    return [
        {"role": "user", "content": "\n\n".join(parts)},
        {"role": "assistant", "content": row["response"]},
    ]


def _fields(row: dict) -> dict:
    return {
        "context": row["context"],
        "instruction": row["instruction"],
        "response": row["response"],
        "summary": row["response"] if row["category"] == "summarization" else None,
    }


def _record(shape: str, rows: list[dict], messages: list[dict]) -> dict:
    return {"messages": messages, "dolly": {"shape": shape, "rows": rows}}


def render_records(pool: list[dict], rng: random.Random) -> list[dict]:
    records, multi = [], []
    for row in map(_fields, pool):
        shape = rng.choices(SHAPES, SHAPE_WEIGHTS)[0]
        if shape == "plain":
            records.append(_record(shape, [row], render_plain(row, rng)))
        elif shape == "document":
            records.append(_record(shape, [row], render_document(row, rng)))
        else:
            multi.append(row)
    rng.shuffle(multi)
    while multi:
        k = rng.choice((2, 3))
        group, multi = multi[:k], multi[k:]
        shape = "multi" if len(group) > 1 else "document"
        records.append(
            _record(shape, group, [m for row in group for m in render_document(row, rng)])
        )
    return records


HELD_OUT_SEED = 0


def partition(rows: list[dict], held_out: int = 50) -> tuple[list[dict], list[dict]]:
    """Fixed seed: the probe set must not move with --seed, or a corpus drawn
    under another seed could carry probe contexts."""
    shuffled = list(rows)
    random.Random(HELD_OUT_SEED).shuffle(shuffled)
    return shuffled[:held_out], shuffled[held_out:]


def _ack_kind(row: dict, ack: str) -> str:
    if row["summary"] is not None:
        return "summary" if ack == row["summary"] else "opener+summary"
    return "restatement" if ack == row["response"] else "opener"


def _log(records: list[dict]) -> None:
    shapes = Counter(rec["dolly"]["shape"] for rec in records)
    print(f"[{ts()}] records per shape: {dict(shapes)}")
    print(
        f"[{ts()}] rows per shape: "
        f"{dict(Counter(r['dolly']['shape'] for r in records for _ in r['dolly']['rows']))}"
    )
    doc_turns = [
        (row, rec["messages"][2 * i], rec["messages"][2 * i + 1])
        for rec in records
        if rec["dolly"]["shape"] != "plain"
        for i, row in enumerate(rec["dolly"]["rows"])
    ]
    prefaced = sum(user["content"] != row["context"] for row, user, _ in doc_turns)
    print(f"[{ts()}] document turns: {len(doc_turns)}, prefaced {prefaced}")
    acks = Counter(_ack_kind(row, asst["content"]) for row, _, asst in doc_turns)
    print(f"[{ts()}] acknowledgements: {dict(acks)}")

    def plain_first(first: str):
        return lambda rec: rec["messages"][0]["content"].startswith(rec["dolly"]["rows"][0][first])

    examples = {
        "plain, context first": lambda rec: (
            rec["dolly"]["shape"] == "plain" and plain_first("context")(rec)
        ),
        "plain, instruction first": lambda rec: (
            rec["dolly"]["shape"] == "plain" and plain_first("instruction")(rec)
        ),
        "document, prefaced": lambda rec: (
            rec["dolly"]["shape"] == "document"
            and rec["messages"][0]["content"] != rec["dolly"]["rows"][0]["context"]
        ),
        "document, bare": lambda rec: (
            rec["dolly"]["shape"] == "document"
            and rec["messages"][0]["content"] == rec["dolly"]["rows"][0]["context"]
        ),
        "multi": lambda rec: rec["dolly"]["shape"] == "multi",
    }
    for name, match in examples.items():
        rec = next((r for r in records if match(r)), None)
        if rec is None:
            print(f"[{ts()}] example {name}: none")
            continue
        for user, asst in zip(rec["messages"][::2], rec["messages"][1::2]):
            print(f"[{ts()}] example {name} user: {user['content'][:400]!r}")
            print(f"[{ts()}] example {name} assistant: {asst['content'][:200]!r}")


def write_rows(rows: list[dict], args: argparse.Namespace) -> None:
    probe, pool = partition(rows)
    rng = random.Random(args.seed)

    held_out = Path(args.held_out_output)
    held_out.parent.mkdir(parents=True, exist_ok=True)
    held_out.write_text(json.dumps([row["context"] for row in probe]))

    drawn = (
        pool if args.max_examples is None else rng.sample(pool, min(args.max_examples, len(pool)))
    )
    records = render_records(drawn, rng)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in records)

    print(f"[{ts()}] {len(rows)} context rows: {len(probe)} probe -> {held_out}, {len(pool)} pool")
    print(f"[{ts()}] wrote {len(records)} records ({len(drawn)} rows) to {out}")
    _log(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Dolly context rows to messages JSONL")
    parser.add_argument(
        "--max-examples", type=int, default=None, help="Pool rows drawn (default: all)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Draw seed (default: %(default)s)")
    parser.add_argument("--output", default="data/dolly_context.jsonl", help="Output JSONL path")
    parser.add_argument(
        "--held-out-output",
        default="data/dolly_heldout.json",
        help="Probe contexts JSON path (default: %(default)s)",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"[{ts()}] loading databricks/databricks-dolly-15k")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    write_rows([row for row in ds if row["category"] in CATEGORIES], args)


if __name__ == "__main__":
    main()
