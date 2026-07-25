"""Three-test solvability filter for the cram/needle slices
(notes/DISCUSSION-20260725-cl-sleep-analysis-and-filter-testc.md 6-7).

Every recall item is scored teacher-forced on its credited span with the
plain backbone -- which *is* the M-ablated model -- under three contexts:

  A (well-posed)      source + cue            -> must answer
  B (memory-required) interference + cue      -> must fail (source absent)
  C (SSM-insufficient) source + interference + cue, in-stream -> must fail

An item that fails any test keeps its tokens and loses only its recall
credit: discarded probes cost probes, not tokens, and their passages stay in
the artifact as carrier/interference. Discard *composition* is the
diagnostic -- mostly-A means the cloze construction is bad, mostly-B means
entity substitution isn't biting.

B and C each cost one pass over the block, not one per item: the cues and
answers are already in the stream in order, so a single teacher-forced pass
reads every item's span at its own position. B's stream is the block with
every item's source span cut out; C's is the block verbatim.

Plus an explicit string check first: if the credited answer (or a word of it)
is already visible in the interference before the cue, the item is dropped
without spending a forward pass on it.

  make filter-items ARGS="--data data/train_cram.pt"

Thresholds are per-token mean log-probs and are guesses until calibrated --
run the pilot in the main session, read the printed sample, then set them.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_cram import _find

VERDICTS = ("pass", "fail_leak", "fail_a", "fail_b", "fail_c", "fail_margin")


def interference_stream(ids: torch.Tensor, items: list[dict]):
    """The block with every item's source span removed, plus a position remap
    into it. Cues and answers stay in place, so one pass over this stream
    scores every item's B test at its own position.

    Deduplicated: a group of cram items shares one source passage, and remap
    would otherwise subtract that span once per member.
    """
    cuts = sorted({(it["source_start"], it["source_end"]) for it in items})
    keep, prev = [], 0
    for a, b in cuts:
        keep.append(ids[prev:a])
        prev = max(prev, b)
    keep.append(ids[prev:])
    stream = torch.cat(keep)

    def remap(pos: int) -> int:
        return pos - sum(min(b, pos) - min(a, pos) for a, b in cuts)

    return stream, remap


def aliases(text: str) -> list[str]:
    """The answer plus any long word of it -- a surname alone gives the answer
    away as surely as the full name does."""
    text = text.strip()
    return sorted({text} | {w.strip(".,;:'\"()") for w in text.split() if len(w) >= 4})


def leaks(stream: torch.Tensor, before: int, credit_text: str, tokenizer) -> bool:
    prefix = stream[:before]
    for alias in aliases(credit_text):
        for form in (alias, " " + alias):
            pat = torch.tensor(tokenizer.encode(form, add_special_tokens=False), dtype=prefix.dtype)
            if len(pat) and _find(prefix, pat):
                return True
    return False


def verdict(a: float, b: float, c: float, args) -> str:
    if a < args.a_min:
        return "fail_a"
    if b > args.b_max:
        return "fail_b"
    if c > args.c_max:
        return "fail_c"
    if a - b < args.min_margin:
        return "fail_margin"
    return "pass"


def filter_dataset(dataset: dict, scorer, tokenizer, args) -> dict:
    """Scores every item, zeroes the recall credit of the failures, and
    records the three log-probs and the verdict on each item. Mutates
    `dataset` in place; returns stats."""
    counts = Counter()
    samples: list[str] = []
    n_blocks = len(dataset["ids"])

    for bi, (ids, recall, items) in enumerate(
            zip(dataset["ids"], dataset["recall_masks"], dataset["items"])):
        if not items:
            continue
        stream, remap = interference_stream(ids, items)
        scored, leaked = [], []
        for it in items:
            (leaked if leaks(stream, remap(it["cue_start"]), it["credit_text"], tokenizer) else scored).append(it)
        for it in leaked:
            it["filter"] = {"verdict": "fail_leak"}
            recall[it["span_start"]:it["span_end"]] = False
            counts["fail_leak"] += 1

        if getattr(args, "leak_only", False):
            for it in scored:
                it["filter"] = {"verdict": "pass"}
                counts["pass"] += 1
            scored = []
        if scored:
            spans = [(it["span_start"], it["span_end"]) for it in scored]
            cs = scorer.span_logprobs(ids, spans, "C")
            bs = scorer.span_logprobs(stream, [(remap(a), remap(b)) for a, b in spans], "B")
            as_ = []
            for it in scored:
                ctx = torch.cat([ids[it["source_start"]:it["source_end"]], ids[it["cue_start"]:it["span_end"]]])
                off = (it["source_end"] - it["source_start"]) - it["cue_start"]
                as_ += scorer.span_logprobs(ctx, [(it["span_start"] + off, it["span_end"] + off)], "A")

            for it, a, b, c in zip(scored, as_, bs, cs):
                v = verdict(a, b, c, args)
                it["filter"] = {"a": round(float(a), 4), "b": round(float(b), 4),
                                "c": round(float(c), 4), "verdict": v}
                counts[v] += 1
                if v != "pass":
                    recall[it["span_start"]:it["span_end"]] = False
                if len(samples) < args.samples:
                    samples.append(
                        f"{v} -- A {a:+.2f} (source+cue) B {b:+.2f} (interference, no source) "
                        f"C {c:+.2f} (in-stream), gap {it['gap']}, entity {it['credit_text']!r}\n"
                        f"    [cue]    ...{tokenizer.decode(ids[max(0, it['cue_start'] - 160):it['cue_end']])}\n"
                        f"    [scored] {tokenizer.decode(ids[it['answer_start']:it['span_start']])}"
                        f"<<{tokenizer.decode(ids[it['span_start']:it['span_end']])}>>")
                    # Printed immediately, not held for the end-of-run report: a
                    # degenerate scoring run should be recognizable at item one,
                    # not after the full pass.
                    print(f"\n  [scored sample {len(samples)}] {samples[-1]}")
        comp = " ".join(f"{v.removeprefix('fail_')} {counts[v]}" for v in VERDICTS[1:] if counts[v])
        print(f"\r  filtered {bi + 1}/{n_blocks} blocks, {sum(counts.values())} items, "
              f"{counts['pass']} kept" + (f" ({comp})" if comp else ""), end="", flush=True)
    print()

    n_items = sum(counts.values())
    stats = {
        "n_items": n_items,
        "verdicts": {v: counts[v] for v in VERDICTS},
        "discard_rate": (n_items - counts["pass"]) / n_items if n_items else 0.0,
    }
    print(f"\nfilter: {n_items} items, {counts['pass']} kept, "
          f"discard rate {100 * stats['discard_rate']:.1f}%")
    for v in VERDICTS[1:]:
        share = 100 * counts[v] / max(n_items - counts["pass"], 1)
        print(f"  {v}: {counts[v]} ({share:.1f}% of discards)")
    print("  (mostly fail_a -> the cloze construction is bad; mostly fail_b -> entity substitution "
          "isn't biting; mostly fail_c -> gaps are too short for the interference to defeat the SSM)")
    return stats


def rescore_dataset(dataset: dict, args) -> dict:
    """Recomputes verdicts and recall credit from the scores a previous
    filter run stored on each item -- no model, no GPU. fail_leak items were
    never scored and stay failed. Mutates `dataset` in place; returns stats."""
    counts = Counter()
    for recall, items in zip(dataset["recall_masks"], dataset["items"]):
        for it in items:
            rec = it.get("filter")
            if rec is None:
                continue
            if rec["verdict"] == "fail_leak":
                counts["fail_leak"] += 1
                continue
            if getattr(args, "leak_only", False):
                v = "pass"
            elif "a" not in rec:
                counts[rec["verdict"]] += 1
                continue
            else:
                v = verdict(rec["a"], rec["b"], rec["c"], args)
            rec["verdict"] = v
            counts[v] += 1
            recall[it["span_start"]:it["span_end"]] = v == "pass"

    n_items = sum(counts.values())
    stats = {
        "n_items": n_items,
        "verdicts": {v: counts[v] for v in VERDICTS},
        "discard_rate": (n_items - counts["pass"]) / n_items if n_items else 0.0,
    }
    print(f"rescore: {n_items} items, {counts['pass']} kept, "
          f"discard rate {100 * stats['discard_rate']:.1f}%")
    for v in VERDICTS[1:]:
        share = 100 * counts[v] / max(n_items - counts["pass"], 1)
        print(f"  {v}: {counts[v]} ({share:.1f}% of discards)")
    return stats


class BackboneScorer:
    """Teacher-forced span scoring with the repo's own chunked state-carrying
    forward (the same pattern as probe_recall.run_chunks), so a 32k-token
    block never materializes 32k rows of vocab logits at once."""

    def __init__(self, model, device, chunk_len: int = 256):
        self.model = model
        self.device = device
        self.chunk_len = chunk_len

    @torch.no_grad()
    def span_logprobs(self, ids, spans: list[tuple[int, int]], label: str) -> list[float]:
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=self.device).view(1, -1)
        totals = [0.0] * len(spans)
        state, prev_row = None, None
        n = ids_t.shape[1]
        for start in range(0, n, self.chunk_len):
            chunk = ids_t[:, start:start + self.chunk_len]
            logits, state = self.model(chunk, state=state)
            state = state.detach()
            logprobs = torch.log_softmax(logits[0].float(), dim=-1)
            for si, (a, b) in enumerate(spans):
                for t in range(max(a, start), min(b, start + chunk.shape[1])):
                    row = logprobs[t - 1 - start] if t - 1 >= start else prev_row
                    totals[si] += float(row[ids_t[0, t]])
            prev_row = logprobs[-1]
        return [totals[si] / max(b - a, 1) for si, (a, b) in enumerate(spans)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="A prepare_cram.py/prepare_needles.py artifact")
    parser.add_argument("--output", default=None, help="Default: <data> with -filtered before the suffix")
    parser.add_argument("--a-min", type=float, default=-2.0,
                        help="Test A passes at or above this mean log-prob per credited token. A loose "
                             "sanity floor only: entity substitution makes the span deliberately "
                             "implausible, so the backbone's prior fights the copy even with the source "
                             "in view (pilot: fail_a median A -1.4 at margins of 6+ nats). The margin "
                             "rule carries the real discrimination")
    parser.add_argument("--b-max", type=float, default=-1.5,
                        help="Test B passes at or below this. -1.5 (~0.22/token) sits just above bAbI's "
                             "~6-way chance level, so closed-vocabulary guessing still passes and the "
                             "margin rule does the real work there")
    parser.add_argument("--c-max", type=float, default=-1.5, help="Test C passes at or below this")
    parser.add_argument("--min-margin", type=float, default=4.0,
                        help="Required A - B in nats/token: the source, not inferability, must carry "
                             "the answer. Scale-free where the absolute thresholds are not: a "
                             "common-word answer scores high everywhere. Margin is over B only -- C has "
                             "its own absolute bar, and requiring a large A-C gap double-counts it, "
                             "killing items that are near-certain with the source and dead in-stream. "
                             "4.0 calibrated on the pilot (pass margins median ~6.4, ill-posed below ~3)")
    parser.add_argument("--leak-only", action="store_true",
                        help="Skip the A/B/C scoring; only the string leak-check gates credit. For the "
                             "needle slice: bAbI items are well-posed and leak-proof by construction, "
                             "and the base backbone cannot express bare-entity answers in the marker "
                             "format at all (pilot: A -12..-18 in every context, so scoring measures "
                             "format competence, not item quality)")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-verdict a previously-scored artifact from its stored per-item scores "
                             "-- no model load, no GPU. Threshold sweeps cost seconds instead of a "
                             "scoring pass")
    parser.add_argument("--chunk-len", type=int, default=256, help="Forward chunk for the scoring pass")
    parser.add_argument("--samples", type=int, default=3, help="Decoded scored items printed")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = torch.load(args.data, map_location="cpu", weights_only=False)
    if args.rescore:
        stats = rescore_dataset(dataset, args)
    else:
        import importlib
        import os

        from models.common import build_tokenizer

        model_name = os.getenv("MODEL_NAME", "mamba2_780m")
        model_mod = importlib.import_module(f"models.{model_name}")
        tokenizer = build_tokenizer(model_mod)
        print(f"loading {model_name} on {args.device} (the plain backbone IS the M-ablated model) ...")
        model = model_mod.load_inference(args.device)
        model.eval()
        stats = filter_dataset(dataset, BackboneScorer(model, args.device, args.chunk_len),
                               tokenizer, args)
    dataset["filter"] = {"thresholds": {"a_min": args.a_min, "b_max": args.b_max,
                                        "c_max": args.c_max, "min_margin": args.min_margin},
                         **stats}
    out = Path(args.output) if args.output else Path(args.data).with_name(
        Path(args.data).stem + "-filtered.pt")
    torch.save(dataset, out)
    n_recall = sum(int(r.sum()) for r in dataset["recall_masks"])
    print(f"\nwrote {out}: {n_recall} recall-credited tokens remain")


if __name__ == "__main__":
    main()
