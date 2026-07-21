"""Builds episodic-chain training data for the three-tier memory design
(docs/superpowers/specs/2026-07-17-episodic-chains-design.md).

A chain is one long training example: a shuffled sequence of episodes
(whole conversations from the source datasets) concatenated up to a sampled
token budget, with three things layered in:

- **Sleeps** (`sleep_positions` metadata): token offsets where train.py
  wipes the slot's backbone state while the neural memory persists.
  Placed at randomly chosen between-episode boundaries so a wake spans
  1-4 episodes (unpredictable -- a fixed cadence would train a pre-sleep
  cramming policy), plus a mid-conversation sleep at a between-turn
  boundary in a fraction of long episodes -- the natural-continuation
  signal: every ordinary token after that wipe is predicted better iff the
  memory retained the gist of what preceded it, a dense training signal
  with no engineered template to overfit to.
- **Interleaved continuations** (`--split-episode-rate`): a fraction of
  eligible episodes are split at a turn boundary (>= --split-min-part
  tokens on each side) and the tail resumes two episodes later behind a
  forced sleep -- the natural-continuation signal stretched across an
  intervening episode, training the memory to retain a conversation's gist
  through unrelated material. For single-QA episodes (a long document turn
  then its answer) the cut lands at the answer boundary: read the document
  now, answer it an episode and a sleep later.
- **Fact blocks**: a fraction of episodes host a block of key/value facts
  (heterogeneous kinds and phrasings, shared with prepare_interference.py),
  spliced at the episode's second turn boundary. A fraction of facts are
  later revised, training delta-rule overwrite.
- **Queries at three distances**: each block gets query/answer pairs whose
  insertion point is drawn from within the same episode, from a later
  episode in the same wake (no sleep between -- the SSM's legitimate job),
  or from beyond a sleep (answerable only through the neural memory).

Everything operates on token ids -- carrier conversations are never
re-tokenized, only the short injected turns are (one batched call).
Fact keys use the vocab-scan slice [1024, 2048), disjoint from
probe_recall.py's [0, 1024), so the probe stays a held-out eval.

Output: {ids, masks, recall_masks, sleep_positions}. recall_masks are True
exactly on spliced answer-content tokens (for --recall-weight);
sleep_positions are sorted post-splice token offsets per chain.

  make prepare-chains     # data/train.pt + data/train_memory.pt -> data/train_chains.pt
  uv run --no-sync python prepare_chains.py --sources data/train_memory.pt
"""

import argparse
import math
import random
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_interference import FACT_KINDS, LABEL_POOL, LABEL_SKIP
from probe_recall import single_token_labels


def sample_log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def build_chains(
    pool_ids: list[torch.Tensor],
    pool_masks: list[torch.Tensor],
    encode,
    labels: list[str],
    user_open: str,
    asst_open: str,
    user_id: int,
    asst_id: int,
    args,
) -> tuple[dict, dict]:
    """The whole generator, IO-free: `encode` is a batched
    strings -> list[list[int]] tokenizer callable, `args` the CLI namespace.
    Returns (dataset dict as written to --output, stats dict)."""

    def sample_value(kind, rng: random.Random) -> str:
        if kind["value"] == "vocab_word":
            return rng.choice(labels)
        return kind["value"](rng)

    rng = random.Random(args.seed)
    order = list(range(len(pool_ids)))
    rng.shuffle(order)

    # Partition the shuffled pool into chains by token budget: append
    # episodes until the sampled budget is crossed, close the chain there.
    chains: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    budget = sample_log_uniform(rng, args.min_budget, args.max_budget)
    for ep in order:
        current.append(ep)
        current_tokens += len(pool_ids[ep])
        if current_tokens >= budget:
            chains.append(current)
            current, current_tokens = [], 0
            budget = sample_log_uniform(rng, args.min_budget, args.max_budget)
    if current:
        chains.append(current)
    print(f"planned {len(chains)} chains from {len(order)} episodes")

    # Interleaved continuations: split an eligible episode at a middle-third
    # turn boundary and resume its tail two episodes later, behind a forced
    # sleep (added in the sleep-placement pass below via split_tails) -- the
    # tail's tokens are predicted better iff the memory carried the head's
    # gist through an intervening episode AND a sleep, so this trains
    # cross-episode retention with no engineered template.
    pool_ids = list(pool_ids)
    pool_masks = list(pool_masks)
    split_tails: set[int] = set()
    n_split = 0
    if getattr(args, "split_episode_rate", 0.0) > 0:
        for ci, chain in enumerate(chains):
            out: list[int] = []
            pending: list[tuple[int, int]] = []  # (due position in out, tail episode)
            for ep in chain:
                ids = pool_ids[ep]
                mp = args.split_min_part
                if len(ids) >= 2 * mp and rng.random() < args.split_episode_rate:
                    b = ((ids == user_id) | (ids == asst_id)).nonzero().flatten().tolist()
                    mid = [x for x in b if mp <= x <= len(ids) - mp]
                    if mid:
                        cut = rng.choice(mid)
                        pool_ids.append(ids[:cut])
                        pool_masks.append(pool_masks[ep][:cut])
                        out.append(len(pool_ids) - 1)
                        pool_ids.append(ids[cut:])
                        pool_masks.append(pool_masks[ep][cut:])
                        split_tails.add(len(pool_ids) - 1)
                        pending.append((len(out) + 2, len(pool_ids) - 1))
                        n_split += 1
                        continue
                out.append(ep)
            # Dues are in pre-insertion coordinates; each earlier-inserted
            # tail shifts later dues by one.
            for i, (due, tail) in enumerate(sorted(pending)):
                out.insert(min(due + i, len(out)), tail)
            chains[ci] = out
    if n_split:
        print(f"split {n_split} episodes into head/tail interleaved continuations")

    # Pass 1: plan every chain -- sleeps, fact blocks, revisions, queries --
    # and collect all injected-turn strings for one batched tokenizer call.
    strings: list[str] = []

    def add_string(s: str) -> int:
        strings.append(s)
        return len(strings) - 1

    plans = []  # per chain: (episode_idxs, inserts, sleep_offsets_presplice)
    n_blocks = n_facts_total = n_revised = n_mid_sleeps = 0
    dist_counts = {"within_episode": 0, "cross_episode": 0, "cross_sleep": 0}
    for chain in chains:
        offsets = [0]
        for ep in chain:
            offsets.append(offsets[-1] + len(pool_ids[ep]))

        # Turn boundaries in chain coordinates, per episode; each episode's
        # first boundary is excluded from insertion candidates so no insert
        # ever ties with a between-episode sleep at the episode's start.
        ep_boundaries: list[list[int]] = []
        for i, ep in enumerate(chain):
            b = ((pool_ids[ep] == user_id) | (pool_ids[ep] == asst_id)).nonzero().flatten()
            ep_boundaries.append([offsets[i] + x for x in b.tolist()[1:]])

        # Between-episode sleeps: wakes of min-wake..max-wake episodes,
        # wake length re-sampled after each sleep so placement is
        # unpredictable.
        sleeps: list[int] = []
        wake = rng.randint(args.min_wake, args.max_wake)
        since_sleep = 0
        for i in range(len(chain) - 1):
            since_sleep += 1
            if since_sleep >= wake:
                sleeps.append(offsets[i + 1])
                since_sleep = 0
                wake = rng.randint(args.min_wake, args.max_wake)
        # Forced sleep at each split tail's start, so resuming the head's
        # conversation always crosses a sleep.
        for i, ep in enumerate(chain):
            if ep in split_tails:
                sleeps.append(offsets[i])
        # Mid-conversation sleeps in long episodes, at a between-turn
        # boundary in the middle third of the episode.
        for i, ep in enumerate(chain):
            if len(pool_ids[ep]) < args.mid_sleep_min_len or rng.random() >= args.mid_sleep_rate:
                continue
            lo = offsets[i] + len(pool_ids[ep]) // 3
            hi = offsets[i] + 2 * len(pool_ids[ep]) // 3
            mid = [b for b in ep_boundaries[i] if lo <= b <= hi]
            if mid:
                sleeps.append(rng.choice(mid))
                n_mid_sleeps += 1
        sleeps = sorted(set(sleeps))
        sleep_set = set(sleeps)
        # An insert exactly at a sleep offset would be ambiguous (does the
        # model see it before or after the wipe?) -- drop those candidates.
        candidates_per_ep = [[b for b in bs if b not in sleep_set] for bs in ep_boundaries]
        all_candidates = sorted(b for bs in candidates_per_ep for b in bs)

        # Fact blocks. Keys are unique across the whole chain (a chain is one
        # continuous life for the memory -- duplicate keys would make two
        # different facts collide at one address).
        chain_labels = rng.sample(labels, len(labels))
        inserts: list[tuple[int, list[int], bool]] = []
        for i, ep in enumerate(chain):
            if rng.random() >= args.fact_rate or len(candidates_per_ep[i]) < 2:
                continue
            n_facts = int(sample_log_uniform(rng, args.min_facts, args.max_facts))
            if len(chain_labels) < max(n_facts, args.min_facts):
                continue
            n_facts = min(n_facts, len(chain_labels))
            block_keys, chain_labels = chain_labels[:n_facts], chain_labels[n_facts:]
            block_pos = candidates_per_ep[i][0]
            n_blocks += 1
            n_facts_total += n_facts

            fact_turn_idx = []
            facts = []
            for k in block_keys:
                kind = rng.choice(FACT_KINDS)
                stmt, q, a, rev = rng.choice(kind["sets"])
                v = sample_value(kind, rng)
                fact = {"templates": (q, a, rev), "kind": kind, "k": k, "v": v, "v2": None, "floor": block_pos}
                fact_turn_idx.append(add_string(f"{user_open} {stmt.format(k=k, v=v)}"))
                facts.append(fact)
            inserts.append((block_pos, fact_turn_idx, False))

            # Revisions land in the earlier half of the chain's remaining
            # candidates, so post-revision queries still have room after.
            for f in facts:
                if rng.random() >= args.revise_rate:
                    continue
                later = all_candidates[bisect_right(all_candidates, block_pos):]
                if not later:
                    continue
                f["v2"] = sample_value(f["kind"], rng)
                pos = rng.choice(later[: max(1, len(later) // 2)])
                f["floor"] = pos
                inserts.append((pos, [add_string(f"{user_open} {f['templates'][2].format(k=f['k'], v=f['v2'])}")], False))
                n_revised += 1

            # Queries, each at one of three distances from its fact's floor
            # (statement block, or the revision if revised): same episode,
            # later episode within the same wake, or beyond a sleep.
            n_queries = rng.randint(args.min_queries, min(args.max_queries, n_facts))
            for f in rng.sample(facts, n_queries):
                later = all_candidates[bisect_right(all_candidates, f["floor"]):]
                if not later:
                    continue
                by_dist: dict[str, list[int]] = {"within_episode": [], "cross_episode": [], "cross_sleep": []}
                for b in later:
                    if bisect_right(sleeps, b) > bisect_right(sleeps, f["floor"]):
                        by_dist["cross_sleep"].append(b)
                    elif offsets[i] <= b < offsets[i + 1]:
                        by_dist["within_episode"].append(b)
                    else:
                        by_dist["cross_episode"].append(b)
                # Only cross_sleep queries actually require the neural memory
                # (a sleep wipes the SSM between the fact and the query); the
                # other two distances are SSM-answerable within the wake. With
                # uniform selection the memory-requiring signal is only ~1/3 of
                # the queries, so --cross-sleep-bias biases toward it when the
                # query has a cross_sleep option (0.0 = uniform, backward-compat).
                if by_dist["cross_sleep"] and rng.random() < args.cross_sleep_bias:
                    dist = "cross_sleep"
                else:
                    dist = rng.choice([d for d, bs in by_dist.items() if bs])
                dist_counts[dist] += 1
                q, a, _ = f["templates"]
                final_v = f["v2"] if f["v2"] is not None else f["v"]
                inserts.append((
                    rng.choice(by_dist[dist]),
                    [add_string(f"{user_open} {q.format(k=f['k'])}"),
                     add_string(f"{asst_open} {a.format(k=f['k'], v=final_v)}")],
                    True,
                ))

        inserts.sort(key=lambda x: x[0])
        plans.append((chain, inserts, sleeps))

    print(f"tokenizing {len(strings)} injected turns "
          f"({n_blocks} fact blocks, {n_facts_total} facts, {n_revised} revisions; "
          f"queries {dist_counts}) ...")
    encoded = encode(strings) if strings else []

    def turn_tensors(idx: int, trainable: bool) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.tensor(encoded[idx], dtype=torch.long)
        m = torch.zeros(len(t), dtype=torch.bool)
        if trainable:
            m[1:] = True  # True on assistant content, False on the role marker itself
        return t, m

    # Pass 2: materialize each chain -- concatenate episodes, splice inserts,
    # and shift each sleep by the length of everything inserted before it.
    out_ids: list[torch.Tensor] = []
    out_masks: list[torch.Tensor] = []
    out_recall: list[torch.Tensor | None] = []
    out_sleeps: list[torch.Tensor] = []
    for done, (chain, inserts, sleeps) in enumerate(plans):
        ids = torch.cat([pool_ids[ep] for ep in chain])
        masks = torch.cat([pool_masks[ep] for ep in chain])

        id_parts, mask_parts, recall_parts, cursor = [], [], [], 0
        insert_positions: list[int] = []
        insert_lengths: list[int] = []
        for pos, turn_ids, is_query in inserts:
            id_parts.append(ids[cursor:pos])
            mask_parts.append(masks[cursor:pos])
            recall_parts.append(torch.zeros(pos - cursor, dtype=torch.bool))
            total = 0
            for j, ti in enumerate(turn_ids):
                t, m = turn_tensors(ti, trainable=(is_query and j == 1))
                id_parts.append(t)
                mask_parts.append(m)
                recall_parts.append(m)  # True exactly on spliced answer content
                total += len(t)
            insert_positions.append(pos)
            insert_lengths.append(total)
            cursor = pos
        id_parts.append(ids[cursor:])
        mask_parts.append(masks[cursor:])
        recall_parts.append(torch.zeros(len(ids) - cursor, dtype=torch.bool))

        shifted = [
            s + sum(insert_lengths[: bisect_left(insert_positions, s)])
            for s in sleeps
        ]
        out_ids.append(torch.cat(id_parts))
        out_masks.append(torch.cat(mask_parts))
        out_recall.append(torch.cat(recall_parts) if inserts else None)
        out_sleeps.append(torch.tensor(shifted, dtype=torch.long))
        if done % 50 == 0:
            print(f"\r  built {done + 1}/{len(plans)} chains", end="", flush=True)
    print(f"\r  built {len(plans)}/{len(plans)} chains")

    dataset = {"ids": out_ids, "masks": out_masks, "recall_masks": out_recall, "sleep_positions": out_sleeps}
    stats = {
        "n_blocks": n_blocks, "n_facts": n_facts_total, "n_revised": n_revised,
        "dist_counts": dist_counts, "n_split": n_split, "n_mid_sleeps": n_mid_sleeps,
    }
    return dataset, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sources", nargs="+", default=["data/train.pt", "data/train_memory.pt"],
                        help="Episode pools (prepare_data.py/merge_data.py outputs); every episode is used exactly once")
    parser.add_argument("--output", default="data/train_chains.pt")
    parser.add_argument("--min-budget", type=int, default=30_000, help="Per-chain token budget, log-uniform lower bound")
    parser.add_argument("--max-budget", type=int, default=130_000, help="Per-chain token budget, log-uniform upper bound")
    parser.add_argument("--min-wake", type=int, default=1, help="Minimum episodes per wake (between sleeps)")
    parser.add_argument("--max-wake", type=int, default=4, help="Maximum episodes per wake")
    parser.add_argument("--mid-sleep-rate", type=float, default=0.2,
                        help="Fraction of long episodes that get one mid-conversation sleep (the natural-continuation signal)")
    parser.add_argument("--split-episode-rate", type=float, default=0.0,
                        help="Fraction of eligible episodes split at a turn boundary (>= --split-min-part tokens on each side) with the tail resumed two episodes later behind a forced sleep -- trains cross-episode gist retention (interleaved continuation)")
    parser.add_argument("--split-min-part", type=int, default=256,
                        help="Minimum tokens on each side of a split-episode cut boundary")
    parser.add_argument("--mid-sleep-min-len", type=int, default=4096,
                        help="Minimum episode length in tokens to be eligible for a mid-conversation sleep")
    parser.add_argument("--fact-rate", type=float, default=0.3, help="Fraction of episodes that host a fact block")
    parser.add_argument("--min-facts", type=int, default=4)
    parser.add_argument("--max-facts", type=int, default=64)
    parser.add_argument("--min-queries", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--revise-rate", type=float, default=0.12, help="Fraction of facts later revised to a new value")
    parser.add_argument("--cross-sleep-bias", type=float, default=0.0,
                        help="Probability of forcing a query to cross_sleep distance when that option exists (0.0 = uniform over available distances; only cross_sleep queries require the neural memory)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import models.mamba2_2_7b_memory as model_mod
    from models.common import build_tokenizer

    tokenizer = build_tokenizer(model_mod)
    labels = single_token_labels(tokenizer, LABEL_POOL, skip=LABEL_SKIP)

    pool_ids: list[torch.Tensor] = []
    pool_masks: list[torch.Tensor] = []
    for src in args.sources:
        data = torch.load(src, map_location="cpu", weights_only=True)
        pool_ids.extend(data["ids"])
        pool_masks.extend(data["masks"])
        print(f"  {src}: +{len(data['ids'])} episodes")

    dataset, stats = build_chains(
        pool_ids, pool_masks,
        encode=lambda strings: tokenizer(strings, add_special_tokens=False)["input_ids"],
        labels=labels,
        user_open=model_mod.USER_OPEN, asst_open=model_mod.ASST_OPEN,
        user_id=tokenizer.convert_tokens_to_ids(model_mod.USER_OPEN),
        asst_id=tokenizer.convert_tokens_to_ids(model_mod.ASST_OPEN),
        args=args,
    )

    torch.save(dataset, args.output)
    total = sum(len(t) for t in dataset["ids"])
    n_sleep = sum(len(s) for s in dataset["sleep_positions"])
    n_recall = sum(int(r.sum()) for r in dataset["recall_masks"] if r is not None)
    print(
        f"wrote {args.output}: {len(dataset['ids'])} chains, {total / 1e6:.1f}M tokens, "
        f"{n_sleep} sleeps ({stats['n_mid_sleeps']} mid-conversation, {stats['n_split']} split-tail, "
        f"{stats['n_blocks']} fact blocks, {stats['n_facts']} facts, "
        f"{stats['n_revised']} revisions, queries {stats['dist_counts']}), "
        f"{n_recall / 1e3:.1f}k recall-answer tokens"
    )


if __name__ == "__main__":
    main()
