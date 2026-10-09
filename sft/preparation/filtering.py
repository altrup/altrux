"""Three-test solvability filter for the cram/needle slices
(notes/discussion/DISCUSSION-20260725-cl-sleep-analysis-and-filter-testc.md 6-7).

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

Blocks are independent streams and A's contexts are independent sequences, so
each pass runs --score-batch of them at once, right-padded to the batch's
longest (see pad_rows).

Plus an explicit string check first: if the credited answer (or a word of it)
is already visible in the interference before the cue, the item is dropped
without spending a forward pass on it.

  make filter-items ARGS="--data data/train_cram.pt"

Thresholds are per-token mean log-probs and are guesses until calibrated --
run the pilot in the main session, read the printed sample, then set them.
"""

import argparse
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preparation.cram import find_subsequence

VERDICTS = ("pass", "fail_leak", "fail_a", "fail_b", "fail_c", "fail_margin")


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def _hms(seconds: float) -> str:
    s = max(int(seconds), 0)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


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
            if len(pat) and find_subsequence(prefix, pat):
                return True
    return False


Row = tuple[torch.Tensor, list[tuple[int, int]]]


def length_batches(lengths: list[int], max_rows: int, max_ratio: float = 2.0) -> list[list[int]]:
    """Row indices grouped into batches: sorted by length, then cut whenever
    the batch is full or its longest row would exceed `max_ratio` times its
    shortest. Sorting plus the ratio cap is what bounds the padding waste on
    a long-tailed length distribution."""
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in sorted(range(len(lengths)), key=lengths.__getitem__):
        if cur and (len(cur) >= max_rows or lengths[i] > max_ratio * lengths[cur[0]]):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def pad_rows(rows: list[Row], device) -> tuple[torch.Tensor, list[int]]:
    """Right-pad a batch of rows to its longest, returning the true lengths.

    The backbone's recurrence is causal, so what is fed after a row's own
    tokens cannot reach the log-probs at positions before them -- padding is
    inert exactly as long as every read stays inside the row it belongs to,
    which the span assert enforces.
    """
    lens = [len(ids) for ids, _ in rows]
    padded = torch.zeros(len(rows), max(lens), dtype=torch.long, device=device)
    for i, (ids, spans) in enumerate(rows):
        padded[i, : lens[i]] = torch.as_tensor(ids, dtype=torch.long, device=device)
        for a, b in spans:
            assert 0 < a < b <= lens[i], f"span ({a}, {b}) outside row {i} of length {lens[i]}"
    return padded, lens


def score_rows(
    scorer, rows: list[Row], label: str, max_rows: int, max_ratio: float = 2.0
) -> tuple[list[list[float]], int]:
    """Scores `rows` in padded batches, results in input order. Also returns
    the real token count fed, padding excluded, for the throughput line."""
    out: list[list[float]] = [[] for _ in rows]
    fed = 0
    for idx in length_batches([len(ids) for ids, _ in rows], max_rows, max_ratio):
        for i, row_scores in zip(idx, scorer.span_logprobs([rows[i] for i in idx], label)):
            out[i] = row_scores
        fed += sum(len(rows[i][0]) for i in idx)
    return out, fed


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
    max_rows = max(1, getattr(args, "score_batch", 1))
    total_ids = sum(len(ids) for ids in dataset["ids"])
    t0 = time.time()
    done_blocks = done_ids = fed = 0
    width = 0

    def progress() -> None:
        nonlocal width
        comp = " ".join(f"{v.removeprefix('fail_')} {counts[v]}" for v in VERDICTS[1:] if counts[v])
        elapsed = time.time() - t0
        line = (
            f"{_ts()}   filtered {done_blocks}/{n_blocks} blocks, {sum(counts.values())} items, "
            f"{counts['pass']} kept" + (f" ({comp})" if comp else "")
        )
        if fed and done_ids:
            line += (
                f", {fed / elapsed:,.0f} tok/s, "
                f"ETA {_hms(elapsed * (total_ids - done_ids) / done_ids)}"
            )
        width = max(width, len(line))
        print(f"\r{line:<{width}}", end="", flush=True)

    for w0 in range(0, n_blocks, max_rows):
        prepared = []
        for bi in range(w0, min(w0 + max_rows, n_blocks)):
            ids, recall, items = (
                dataset["ids"][bi],
                dataset["recall_masks"][bi],
                dataset["items"][bi],
            )
            done_blocks, done_ids = bi + 1, done_ids + len(ids)
            if not items:
                continue
            stream, remap = interference_stream(ids, items)
            scored, leaked = [], []
            for it in items:
                (
                    leaked
                    if leaks(stream, remap(it["cue_start"]), it["credit_text"], tokenizer)
                    else scored
                ).append(it)
            for it in leaked:
                it["filter"] = {"verdict": "fail_leak"}
                recall[it["span_start"] : it["span_end"]] = False
                counts["fail_leak"] += 1

            if getattr(args, "leak_only", False):
                for it in scored:
                    it["filter"] = {"verdict": "pass"}
                    counts["pass"] += 1
                scored = []
            if scored:
                prepared.append((ids, recall, stream, remap, scored))

        if prepared:
            # One batch of blocks per pass: B and C are whole independent
            # streams, A a short manufactured context per item.
            spans = [[(it["span_start"], it["span_end"]) for it in sc] for *_, sc in prepared]
            cs, n = score_rows(
                scorer, [(p[0], sp) for p, sp in zip(prepared, spans)], "C", max_rows
            )
            fed += n
            progress()
            bs, n = score_rows(
                scorer,
                [(p[2], [(p[3](a), p[3](b)) for a, b in sp]) for p, sp in zip(prepared, spans)],
                "B",
                max_rows,
            )
            fed += n
            progress()

            a_rows: list[Row] = []
            for ids, _, _, _, sc in prepared:
                for it in sc:
                    ctx = torch.cat(
                        [
                            ids[it["source_start"] : it["source_end"]],
                            ids[it["cue_start"] : it["span_end"]],
                        ]
                    )
                    off = (it["source_end"] - it["source_start"]) - it["cue_start"]
                    a_rows.append((ctx, [(it["span_start"] + off, it["span_end"] + off)]))
            as_, n = score_rows(scorer, a_rows, "A", max_rows)
            fed += n

            k = 0
            for (ids, recall, _, _, sc), b_row, c_row in zip(prepared, bs, cs):
                for it, b, c in zip(sc, b_row, c_row):
                    a, k = as_[k][0], k + 1
                    v = verdict(a, b, c, args)
                    it["filter"] = {
                        "a": round(float(a), 4),
                        "b": round(float(b), 4),
                        "c": round(float(c), 4),
                        "verdict": v,
                    }
                    counts[v] += 1
                    if v != "pass":
                        recall[it["span_start"] : it["span_end"]] = False
                    if len(samples) < args.samples:
                        samples.append(
                            f"{v} -- A {a:+.2f} (source+cue) B {b:+.2f} (interference, no source) "
                            f"C {c:+.2f} (in-stream), gap {it['gap']}, entity {it['credit_text']!r}\n"
                            f"    [cue]    ...{tokenizer.decode(ids[max(0, it['cue_start'] - 160) : it['cue_end']])}\n"
                            f"    [scored] {tokenizer.decode(ids[it['answer_start'] : it['span_start']])}"
                            f"<<{tokenizer.decode(ids[it['span_start'] : it['span_end']])}>>"
                        )
                        # Printed immediately, not held for the end-of-run report: a
                        # degenerate scoring run should be recognizable at batch one,
                        # not after the full pass.
                        print(f"\n{_ts()}   [scored sample {len(samples)}] {samples[-1]}")
        progress()
    print()

    n_items = sum(counts.values())
    stats = {
        "n_items": n_items,
        "verdicts": {v: counts[v] for v in VERDICTS},
        "discard_rate": (n_items - counts["pass"]) / n_items if n_items else 0.0,
    }
    print(
        f"\n{_ts()} filter: {n_items} items, {counts['pass']} kept, "
        f"discard rate {100 * stats['discard_rate']:.1f}%"
    )
    for v in VERDICTS[1:]:
        share = 100 * counts[v] / max(n_items - counts["pass"], 1)
        print(f"{_ts()}   {v}: {counts[v]} ({share:.1f}% of discards)")
    print(
        f"{_ts()}   (mostly fail_a -> the cloze construction is bad; mostly fail_b -> entity substitution "
        "isn't biting; mostly fail_c -> gaps are too short for the interference to defeat the SSM)"
    )
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
            recall[it["span_start"] : it["span_end"]] = v == "pass"

    n_items = sum(counts.values())
    stats = {
        "n_items": n_items,
        "verdicts": {v: counts[v] for v in VERDICTS},
        "discard_rate": (n_items - counts["pass"]) / n_items if n_items else 0.0,
    }
    print(
        f"{_ts()} rescore: {n_items} items, {counts['pass']} kept, "
        f"discard rate {100 * stats['discard_rate']:.1f}%"
    )
    for v in VERDICTS[1:]:
        share = 100 * counts[v] / max(n_items - counts["pass"], 1)
        print(f"{_ts()}   {v}: {counts[v]} ({share:.1f}% of discards)")
    return stats


class BackboneScorer:
    """Teacher-forced span scoring with the repo's own chunked state-carrying
    forward (the same pattern as diagnostics.recall.run_chunks), so a 32k-token
    block never materializes 32k rows of vocab logits at once.

    Scores a batch of independent rows at once (see pad_rows). Chunk
    boundaries are absolute positions, identical for every row and identical
    to what a one-row call would use, so batching moves nothing but the batch
    dimension.
    """

    def __init__(self, model, device, chunk_len: int = 256):
        self.model = model
        self.device = device
        self.chunk_len = chunk_len

    @torch.no_grad()
    def span_logprobs(self, rows: list[Row], label: str) -> list[list[float]]:
        ids_t, _ = pad_rows(rows, self.device)
        totals = [[0.0] * len(spans) for _, spans in rows]
        state, prev = None, None
        for start in range(0, ids_t.shape[1], self.chunk_len):
            chunk = ids_t[:, start : start + self.chunk_len]
            logits, state = self.model(chunk, state=state)
            state = state.detach()
            need = [
                (ri, si, t)
                for ri, (_, spans) in enumerate(rows)
                for si, (a, b) in enumerate(spans)
                for t in range(max(a, start), min(b, start + chunk.shape[1]))
            ]
            if need:
                # Credited spans are a few tokens each, so the softmax runs on
                # the handful of rows actually read: a whole-chunk fp32
                # log_softmax is ~50 MB per batch row at this vocab.
                src = torch.stack(
                    [prev[ri] if t == start else logits[ri, t - 1 - start] for ri, _, t in need]
                )
                tgt = ids_t[[ri for ri, _, _ in need], [t for _, _, t in need]]
                vals = (
                    torch.log_softmax(src.float(), dim=-1)
                    .gather(1, tgt.view(-1, 1))
                    .squeeze(1)
                    .tolist()
                )
                for (ri, si, _), v in zip(need, vals):
                    totals[ri][si] += v
            prev = logits[:, -1].clone()
        return [
            [tot / max(b - a, 1) for tot, (a, b) in zip(totals[ri], spans)]
            for ri, (_, spans) in enumerate(rows)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data", required=True, help="A preparation/cram.py/preparation/needles.py artifact"
    )
    parser.add_argument(
        "--output", default=None, help="Default: <data> with -filtered before the suffix"
    )
    parser.add_argument(
        "--a-min",
        type=float,
        default=-2.0,
        help="Test A passes at or above this mean log-prob per credited token. A loose "
        "sanity floor only: entity substitution makes the span deliberately "
        "implausible, so the backbone's prior fights the copy even with the source "
        "in view (pilot: fail_a median A -1.4 at margins of 6+ nats). The margin "
        "rule carries the real discrimination",
    )
    parser.add_argument(
        "--b-max",
        type=float,
        default=-1.5,
        help="Test B passes at or below this. -1.5 (~0.22/token) sits just above bAbI's "
        "~6-way chance level, so closed-vocabulary guessing still passes and the "
        "margin rule does the real work there",
    )
    parser.add_argument("--c-max", type=float, default=-1.5, help="Test C passes at or below this")
    parser.add_argument(
        "--min-margin",
        type=float,
        default=4.0,
        help="Required A - B in nats/token: the source, not inferability, must carry "
        "the answer. Scale-free where the absolute thresholds are not: a "
        "common-word answer scores high everywhere. Margin is over B only -- C has "
        "its own absolute bar, and requiring a large A-C gap double-counts it, "
        "killing items that are near-certain with the source and dead in-stream. "
        "4.0 calibrated on the pilot (pass margins median ~6.4, ill-posed below ~3)",
    )
    parser.add_argument(
        "--leak-only",
        action="store_true",
        help="Skip the A/B/C scoring; only the string leak-check gates credit. For the "
        "needle slice: bAbI items are well-posed and leak-proof by construction, "
        "and the base backbone cannot express bare-entity answers in the marker "
        "format at all (pilot: A -12..-18 in every context, so scoring measures "
        "format competence, not item quality)",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-verdict a previously-scored artifact from its stored per-item scores "
        "-- no model load, no GPU. Threshold sweeps cost seconds instead of a "
        "scoring pass",
    )
    parser.add_argument(
        "--chunk-len", type=int, default=256, help="Forward chunk for the scoring pass"
    )
    parser.add_argument(
        "--score-batch",
        type=int,
        default=16,
        help="Rows per forward: blocks for tests B and C, item contexts for A. Costs "
        "one carried SSM state (~39 MB on the 780m backbone) plus one chunk of "
        "logits (~26 MB at --chunk-len 256) per row, so 16 is ~1 GB above the "
        "model; raise it on a large card. 1 restores the serial path exactly",
    )
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
        print(
            f"{_ts()} loading {model_name} on {args.device} (the plain backbone IS the M-ablated model) ..."
        )
        model = model_mod.load_inference(args.device)
        model.eval()
        stats = filter_dataset(
            dataset, BackboneScorer(model, args.device, args.chunk_len), tokenizer, args
        )
    dataset["filter"] = {
        "thresholds": {
            "a_min": args.a_min,
            "b_max": args.b_max,
            "c_max": args.c_max,
            "min_margin": args.min_margin,
        },
        **stats,
    }
    out = (
        Path(args.output)
        if args.output
        else Path(args.data).with_name(Path(args.data).stem + "-filtered.pt")
    )
    torch.save(dataset, out)
    n_recall = sum(int(r.sum()) for r in dataset["recall_masks"])
    print(f"\n{_ts()} wrote {out}: {n_recall} recall-credited tokens remain")


__all__ = [
    "VERDICTS",
    "Row",
    "interference_stream",
    "aliases",
    "leaks",
    "length_batches",
    "pad_rows",
    "score_rows",
    "verdict",
    "filter_dataset",
    "rescore_dataset",
    "BackboneScorer",
    "main",
]


if __name__ == "__main__":
    main()
