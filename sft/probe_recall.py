"""Recall probe: measures whether the Titans neural memory actually carries
information across long token gaps, or is decorative.

Builds synthetic conversations in the training format: a user turn stating a
random digit code, a long stretch of unrelated filler turns, then a query
turn asking for the code back. Scores the teacher-forced log-prob of the
correct code tokens after the query under three conditions:

  intact   -- normal forward, full carried state.
  ablated  -- identical prefix, but state.neural_memory is replaced with a
              fresh random init right before the query. Mamba's own SSM
              state still carries context, so (intact - ablated) isolates
              the neural memory's specific contribution.
  floor    -- query with no prefix at all (the code's prior given only the
              question), as a sanity floor for both numbers.

A clearly positive (intact - ablated) at long gaps means the memory recalls;
~0 means it is currently non-functional for recall. Codes are random digit
strings, so the floor is ~uniform and any prior leakage is visible.

--sleep adds two cross-sleep conditions per combo: after the full prefix
(facts + gap), the backbone state is wiped via Model.sleep_slot -- the
training regime's sleep -- and the query runs in the fresh wake:

  sleep-intact  -- backbone wiped, neural memory kept. Recall here can only
                   flow through the memory; this is the direct measure of
                   the episodic tier working.
  sleep-ablated -- backbone wiped AND memory replaced fresh. Should sit at
                   the floor; if it doesn't, something other than SSM state
                   or the memory is leaking the answer.

--n-facts sweeps interference: each conversation states that many labeled
codes ("The code for river is 4 8 2 1 3.") and queries one label at random.
One salient fact sits comfortably in Mamba's SSM state, so the memory only
has a measurable job once the fact load exceeds SSM capacity -- sweep until
ablated recall degrades and see whether intact holds up.

Run via `make probe-recall` (defaults) or directly:
  uv run --no-sync python probe_recall.py --gaps 1024 --n-facts 64,128,256
"""

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

CHUNK_LEN = 24

FILLER_SENTENCES = [
    "The weather in the valley stayed mild for most of the season.",
    "A good soup starts with onions cooked slowly until they turn golden.",
    "The train from the coast arrives twice a day, once at dawn and once at dusk.",
    "Most of the library's east wing is dedicated to maritime history.",
    "She repainted the fence a pale shade of green last spring.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Migrating birds tend to follow the river south this time of year.",
    "The old mill has been converted into a small museum of local crafts.",
    "He prefers cycling to work when the mornings are dry.",
    "The committee meets on the first Tuesday of every month.",
    "Fresh basil loses its aroma quickly once the leaves are bruised.",
    "The lighthouse keeper kept meticulous logs of every passing storm.",
    "Their garden produces more zucchini than the whole street can eat.",
    "The concert hall's acoustics favor the string section.",
    "A thin layer of fog settled over the harbor before sunrise.",
    "The bakery on the corner sells out of rye bread by nine.",
]


def single_token_labels(tokenizer, n: int, skip: int = 0) -> list[str]:
    """Deterministic list of n distinct label words that each tokenize to
    exactly one token after a space (scanned from the tokenizer's own vocab),
    so query rows stay equal-length no matter which label each row asks for.
    `skip` offsets into the scan order: the probe uses skip=0 and
    prepare_interference.py trains on a disjoint pool (skip=1024), keeping
    this probe an honest held-out eval."""
    import re

    labels = []
    for tok_id in range(len(tokenizer)):
        piece = tokenizer.decode([tok_id])
        if re.fullmatch(r" [a-z]{4,9}", piece):
            labels.append(piece[1:])
            if len(labels) == skip + n:
                return labels[skip:]
    raise ValueError(f"only found {len(labels)} single-token labels, need {skip + n}")


def build_probe_rows(
    tokenizer, user_open: str, asst_open: str, n_probes: int, n_facts: int, gap_tokens: int, rng: random.Random
):
    """Returns (prefix_ids, query_ids, target_ids), each (n_probes, L) --
    equal lengths across rows by construction: labels are shared across rows
    and single-token, codes are fixed-count single digits (random per row),
    and each row queries one of its labels at random."""
    labels = single_token_labels(tokenizer, n_facts)
    codes = [[[rng.randrange(10) for _ in range(5)] for _ in range(n_facts)] for _ in range(n_probes)]

    fact_rows = []
    for r in range(n_probes):
        ids: list[int] = []
        for f, label in enumerate(labels):
            cs = " " + " ".join(str(d) for d in codes[r][f])
            role = user_open if f % 2 == 0 else asst_open
            ids.extend(tokenizer(f"{role} The code for {label} is{cs}.", add_special_tokens=False)["input_ids"])
        fact_rows.append(ids)
    assert len({len(r) for r in fact_rows}) == 1, "fact rows must tokenize to equal lengths"

    filler_ids: list[int] = []
    i = 0
    while len(filler_ids) < gap_tokens + CHUNK_LEN:
        role = user_open if i % 2 == 0 else asst_open
        filler_ids.extend(
            tokenizer(f"{role} {FILLER_SENTENCES[i % len(FILLER_SENTENCES)]}", add_special_tokens=False)["input_ids"]
        )
        i += 1

    prefix_len = len(fact_rows[0]) + gap_tokens
    prefix_len -= prefix_len % CHUNK_LEN
    fill = filler_ids[: prefix_len - len(fact_rows[0])]
    prefix = torch.tensor([row + fill for row in fact_rows], dtype=torch.long)

    query_rows, target_rows = [], []
    for r in range(n_probes):
        j = rng.randrange(n_facts)
        cs = " " + " ".join(str(d) for d in codes[r][j])
        query_rows.append(
            tokenizer(
                f"{user_open} What was the code for {labels[j]}?{asst_open} The code for {labels[j]} is",
                add_special_tokens=False,
            )["input_ids"]
        )
        target_rows.append(tokenizer(cs, add_special_tokens=False)["input_ids"])
    assert len({len(r) for r in query_rows}) == 1, "query rows must tokenize to equal lengths"
    assert len({len(r) for r in target_rows}) == 1
    query = torch.tensor(query_rows, dtype=torch.long)
    target = torch.tensor(target_rows, dtype=torch.long)
    return prefix, query, target


def clone_state(mmod, state):
    """Deep, storage-independent copy of a MemoryState so two continuations
    can diverge from one prefix without sharing mutable buffers."""
    nm_src = state.neural_memory
    nm = mmod._NeuralMemory.__new__(mmod._NeuralMemory)
    for name in ("w1", "b1", "w2", "b2", "w1_init", "w2_init"):
        setattr(nm, name, getattr(nm_src, name).detach().clone())
    nm.momentum = [s.detach().clone() for s in nm_src.momentum]
    return mmod.MemoryState(
        conv_states=[c.detach().clone() for c in state.conv_states],
        ssm_states=[s.detach().clone() for s in state.ssm_states],
        neural_memory=nm,
        last_o_t=state.last_o_t.detach().clone(),
        last_surprise=state.last_surprise.detach().clone(),
    )


def run_chunks(model, ids: torch.Tensor, state, label: str, keep_logits: bool = True):
    """Forward `ids` in CHUNK_LEN chunks, threading and detaching state.
    Returns (all_logits, final_state); prints live progress. Pass
    keep_logits=False when only the final state matters (the prefix pass):
    accumulated full-vocab logits for a long prefix at many probes are
    several GB and have OOMed a real run."""
    logits_parts = []
    n_chunks = (ids.shape[1] + CHUNK_LEN - 1) // CHUNK_LEN
    for ci in range(n_chunks):
        chunk = ids[:, ci * CHUNK_LEN : (ci + 1) * CHUNK_LEN]
        if chunk.shape[1] % CHUNK_LEN != 0:  # pad tail; padded logits are discarded by the caller
            pad = torch.zeros(ids.shape[0], CHUNK_LEN - chunk.shape[1], dtype=torch.long, device=ids.device)
            chunk = torch.cat([chunk, pad], dim=1)
        logits, state = model(chunk, state=state)
        state = state.detach()
        if keep_logits:
            logits_parts.append(logits.detach())
        print(f"\r  {label}: chunk {ci + 1}/{n_chunks}", end="", flush=True)
    print()
    if not keep_logits:
        return None, state
    return torch.cat(logits_parts, dim=1)[:, : ids.shape[1]], state


def score_targets(model, query: torch.Tensor, target: torch.Tensor, state, label: str) -> torch.Tensor:
    """Teacher-forced mean log-prob per target token, (n_probes,)."""
    seq = torch.cat([query, target], dim=1)
    logits, _ = run_chunks(model, seq, state, label)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    scores = torch.zeros(seq.shape[0])
    q_len = query.shape[1]
    for t in range(target.shape[1]):
        pos = q_len + t - 1  # logits at pos predict token at pos+1
        scores += logprobs[torch.arange(seq.shape[0]), pos, target[:, t]].cpu()
    return scores / target.shape[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gaps", default="1024", help="Comma-separated filler lengths in tokens")
    parser.add_argument("--n-facts", default="64,128", help="Comma-separated fact counts per conversation (interference sweep)")
    parser.add_argument("--n-probes", type=int, default=8, help="Probe conversations per gap (batched together)")
    parser.add_argument("--sleep", action="store_true", help="Also score the cross-sleep conditions (backbone wiped after the prefix, memory kept vs replaced) -- see the module docstring")
    parser.add_argument("--memory-window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    import models.mamba2_2_7b_memory as model_mod
    from models.common import build_tokenizer
    from models.mamba2_2_7b_memory import model as mmod
    from models.mamba2_2_7b_memory import train_hooks
    from train import latest_checkpoint, load_checkpoint

    # Inference-only probe: write()'s create_graph=True second-order graph
    # exists so training can backprop into the write projections -- pure
    # overhead here, and large enough to OOM when sharing the GPU with a
    # live training run.
    orig_write = mmod._NeuralMemory.write

    def _write_no_graph(self, ks, vs, etas, thetas, alphas, create_graph=True):
        return orig_write(self, ks, vs, etas, thetas, alphas, create_graph=False)

    mmod._NeuralMemory.write = _write_no_graph

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = latest_checkpoint()
    if ckpt is None:
        raise SystemExit("no checkpoint found")
    print(f"probing checkpoint: {ckpt}")

    import json

    lora_cfg = json.loads((ckpt / "lora_config.json").read_text())
    model, _ = train_hooks.setup_training(device, lora_cfg["rank"], lora_cfg["alpha"], 0.0)
    load_checkpoint(model, ckpt)
    model.eval()
    model.set_memory_window(args.memory_window)
    tokenizer = build_tokenizer(model_mod)

    rng = random.Random(args.seed)
    results = []
    with torch.no_grad():
        for gap in (int(g) for g in args.gaps.split(",")):
            for n_facts in (int(f) for f in args.n_facts.split(",")):
                tag = f"gap {gap} x{n_facts}"
                prefix, query, target = build_probe_rows(
                    tokenizer, model_mod.USER_OPEN, model_mod.ASST_OPEN, args.n_probes, n_facts, gap, rng
                )
                prefix, query, target = prefix.to(device), query.to(device), target.to(device)
                print(f"{tag}: prefix {prefix.shape[1]} tokens ({n_facts} facts + {gap} filler), {args.n_probes} probes")

                _, state = run_chunks(model, prefix, None, f"{tag} prefix", keep_logits=False)
                intact = score_targets(model, query, target, clone_state(mmod, state), f"{tag} intact")

                mem_dtype = model.front_end.q_proj.weight.dtype
                abl_state = clone_state(mmod, state)
                abl_state.neural_memory = model.front_end.init_memory(args.n_probes, device, mem_dtype)
                abl_state.last_o_t.zero_()
                abl_state.last_surprise.zero_()
                ablated = score_targets(model, query, target, abl_state, f"{tag} ablated")

                floor = score_targets(model, query, target, None, f"{tag} floor")

                sleep_scores = {}
                if args.sleep:
                    slept = clone_state(mmod, state)
                    for b in range(args.n_probes):
                        model.sleep_slot(slept, b)
                    sleep_scores["sleep-intact"] = score_targets(
                        model, query, target, slept, f"{tag} sleep-intact"
                    )
                    slept_abl = clone_state(mmod, state)
                    for b in range(args.n_probes):
                        model.sleep_slot(slept_abl, b)
                    slept_abl.neural_memory = model.front_end.init_memory(args.n_probes, device, mem_dtype)
                    sleep_scores["sleep-ablated"] = score_targets(
                        model, query, target, slept_abl, f"{tag} sleep-ablated"
                    )

                diff = intact - ablated
                results.append((gap, n_facts, intact, ablated, floor, diff, sleep_scores))
                line = (
                    f"{tag}: intact {intact.mean():.3f}  ablated {ablated.mean():.3f}"
                    f"  floor {floor.mean():.3f}  memory-delta {diff.mean():.3f} (±{diff.std():.3f})"
                )
                if sleep_scores:
                    s_diff = sleep_scores["sleep-intact"] - sleep_scores["sleep-ablated"]
                    line += (
                        f"  sleep-intact {sleep_scores['sleep-intact'].mean():.3f}"
                        f"  sleep-ablated {sleep_scores['sleep-ablated'].mean():.3f}"
                        f"  sleep-delta {s_diff.mean():.3f} (±{s_diff.std():.3f})"
                    )
                print(line)

    print("\nsummary (mean log-prob per code token; higher = better recall):")
    header = f"{'gap':>8} {'facts':>6} {'intact':>8} {'ablated':>8} {'floor':>8} {'mem-delta':>10}"
    if args.sleep:
        header += f" {'slp-int':>8} {'slp-abl':>8} {'slp-delta':>10}"
    print(header)
    for gap, n_facts, intact, ablated, floor, diff, sleep_scores in results:
        row = f"{gap:>8} {n_facts:>6} {intact.mean():>8.3f} {ablated.mean():>8.3f} {floor.mean():>8.3f} {diff.mean():>10.3f}"
        if sleep_scores:
            s_diff = sleep_scores["sleep-intact"] - sleep_scores["sleep-ablated"]
            row += (
                f" {sleep_scores['sleep-intact'].mean():>8.3f}"
                f" {sleep_scores['sleep-ablated'].mean():>8.3f} {s_diff.mean():>10.3f}"
            )
        print(row)


if __name__ == "__main__":
    main()
