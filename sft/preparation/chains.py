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
  through unrelated material. Single-QA episodes (a long document turn then
  its answer) instead cut at the *question start* recorded by
  prepare_data.py (`--split-qa-rate`): the head keeps the document only
  (suspended behind a sleep), and the tail re-emits a fresh [USER] marker +
  " " before the dataset's own question and answer -- read the document
  now, get asked about it an episode and a sleep later, with every content
  token dataset-authored. Episodes without a recorded question never split.
- **Fact blocks**: a fraction of episodes host a block of key/value facts
  (heterogeneous kinds and phrasings, shared with prepare_interference.py),
  spliced at the episode's second turn boundary. A fraction of facts are
  later revised, training delta-rule overwrite.
- **Queries at three distances**: each block gets query/answer pairs whose
  insertion point is drawn from within the same episode, from a later
  episode in the same wake (no sleep between -- the SSM's legitimate job),
  or from beyond a sleep (answerable only through the neural memory).

Only a `--sleep-chain-rate` fraction of chains carries any of that
apparatus; the rest are plain multi-episode concatenations with silent
joins. Chains are the deployment shape -- retention pressure belongs to the
cram slices (notes/DISCUSSION-20260724-next-run-plan.md §1.3).

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from preparation.interference import FACT_KINDS, LABEL_POOL, LABEL_SKIP


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
    sent_end_ids: set[int] | None = None,
    space_start_ids: set[int] | None = None,
    pool_qoffs: list[int | None] | None = None,
    sep_id: int | None = None,
) -> tuple[dict, dict]:
    """The whole generator, IO-free: `encode` is a batched
    strings -> list[list[int]] tokenizer callable, `args` the CLI namespace.
    Returns (dataset dict as written to --output, stats dict).

    sent_end_ids/space_start_ids feed --sentence-sleep-rate: a sentence
    boundary is a sent_end token immediately followed by a space_start token.

    pool_qoffs (aligned with pool_ids) carries each episode's question token
    offset from prepare_data.py, or None; sep_id is the token id of a literal
    " ". Both feed split-QA: single-QA episodes with an offset are cut at the
    question start, and the tail re-emits [user_id, sep_id] before the
    verbatim question tokens. Episodes without an offset never split."""
    sentence_sleep_rate = getattr(args, "sentence_sleep_rate", 0.0)
    sent_end_t = torch.tensor(sorted(sent_end_ids)) if sent_end_ids else None
    space_start_t = torch.tensor(sorted(space_start_ids)) if space_start_ids else None

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

    # Retention pressure lives in the cram slices; chains are deployment
    # shape, so only a fraction of them carry the engineered apparatus
    # (sleeps, splits, fact blocks). The rest are plain concatenations with
    # silent joins. Drawing nothing at rate 1.0 keeps the RNG stream --
    # and so the output -- identical to an ungated run.
    sleep_rate = getattr(args, "sleep_chain_rate", 1.0)
    sleeping = ([True] * len(chains) if sleep_rate >= 1.0
                else [rng.random() < sleep_rate for _ in chains])
    n_sleep_chains = sum(sleeping)
    print(f"planned {len(chains)} chains from {len(order)} episodes "
          f"({n_sleep_chains} sleeping, {100 * n_sleep_chains / max(len(chains), 1):.1f}%)")

    # Interleaved continuations: split an eligible episode at a middle-third
    # turn boundary and resume its tail two episodes later, behind a forced
    # sleep (added in the sleep-placement pass below via split_tails) -- the
    # tail's tokens are predicted better iff the memory carried the head's
    # gist through an intervening episode AND a sleep, so this trains
    # cross-episode retention with no engineered template.
    pool_ids = list(pool_ids)
    pool_masks = list(pool_masks)
    pool_qoffs = list(pool_qoffs) if pool_qoffs is not None else [None] * len(pool_ids)
    split_tails: set[int] = set()
    split_qa_heads: set[int] = set()
    n_split = n_split_qa = max_pending = 0
    qa_rate = getattr(args, "split_qa_rate", None)
    gap_min = getattr(args, "split_gap_min", 2)
    gap_max = getattr(args, "split_gap_max", 2)
    if getattr(args, "split_episode_rate", 0.0) > 0 or (qa_rate or 0.0) > 0:
        for ci, chain in enumerate(chains):
            if not sleeping[ci]:
                continue
            out: list[int] = []
            pending: list[tuple[int, int]] = []  # (due position in out, tail episode)
            for ep in chain:
                ids = pool_ids[ep]
                mp = args.split_min_part
                if len(ids) > mp:
                    b = ((ids == user_id) | (ids == asst_id)).nonzero().flatten().tolist()
                    # Single-QA episodes (two turns: document then answer) cut
                    # at the question start recorded by prepare_data.py: head
                    # keeps the document only, the tail re-emits a fresh user
                    # marker before the dataset's own question + answer. No
                    # recorded question -> never split (fail closed; e.g.
                    # LongAlign, whose question isn't mechanically
                    # extractable, stays whole as carrier data).
                    is_qa = len(b) == 2
                    rate = qa_rate if (is_qa and qa_rate is not None) else args.split_episode_rate
                    if is_qa:
                        # --split-min-part constrains the head (the retained
                        # document) only: a QA tail is the dataset's own
                        # question + answer, ~10 tokens in babilong, and
                        # requiring 256 of them silently disqualified every
                        # episode in the pool.
                        qoff = pool_qoffs[ep]
                        splittable = qoff is not None and sep_id is not None and mp <= qoff < len(ids)
                        mid = [qoff] if splittable else []
                    else:
                        mid = [x for x in b if mp <= x <= len(ids) - mp]
                    if mid and rng.random() < rate:
                        cut = rng.choice(mid)
                        pool_ids.append(ids[:cut])
                        pool_masks.append(pool_masks[ep][:cut])
                        pool_qoffs.append(None)
                        out.append(len(pool_ids) - 1)
                        if is_qa:
                            split_qa_heads.add(len(pool_ids) - 1)
                            tail_ids = torch.cat([torch.tensor([user_id, sep_id], dtype=ids.dtype), ids[cut:]])
                            tail_masks = torch.cat([torch.zeros(2, dtype=torch.bool), pool_masks[ep][cut:]])
                        else:
                            tail_ids = ids[cut:]
                            tail_masks = pool_masks[ep][cut:]
                        pool_ids.append(tail_ids)
                        pool_masks.append(tail_masks)
                        pool_qoffs.append(None)
                        split_tails.add(len(pool_ids) - 1)
                        # How many earlier heads are still awaiting their tail
                        # here -- a deep interleave stacks several unanswered
                        # turns at once, much worse than a single suspension.
                        max_pending = max(max_pending, 1 + sum(1 for due, _ in pending if due > len(out)))
                        pending.append((len(out) + rng.randint(gap_min, gap_max), len(pool_ids) - 1))
                        n_split += 1
                        n_split_qa += is_qa
                        continue
                out.append(ep)
            # Dues are in pre-insertion coordinates; each earlier-inserted
            # tail shifts later dues by one.
            for i, (due, tail) in enumerate(sorted(pending)):
                out.insert(min(due + i, len(out)), tail)
            chains[ci] = out
    if n_split:
        print(f"split {n_split} episodes ({n_split_qa} single-QA) into head/tail interleaved continuations")

    # Pass 1: plan every chain -- sleeps, fact blocks, revisions, queries --
    # and collect all injected-turn strings for one batched tokenizer call.
    strings: list[str] = []

    def add_string(s: str) -> int:
        strings.append(s)
        return len(strings) - 1

    plans = []  # per chain: (episode_idxs, inserts, sleep_offsets_presplice)
    n_blocks = n_facts_total = n_revised = n_mid_sleeps = n_sentence_sleeps = 0
    dist_counts = {"within_episode": 0, "cross_episode": 0, "cross_sleep": 0}
    for chain, sleeps_on in zip(chains, sleeping):
        if not sleeps_on:
            plans.append((chain, [], []))
            continue
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
            # Suspension sleep at a split-QA head's end: the head is a
            # document turn suspended without its question, so the sleep marks
            # the suspension (making the head->next USER->USER adjacency a
            # legal suspension for validate) and forces the document out of
            # the SSM immediately -- retention must live in the memory.
            if ep in split_qa_heads and offsets[i + 1] < offsets[-1]:
                sleeps.append(offsets[i + 1])
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
        # Sentence-boundary sleeps: for long episodes whose middle third has
        # no turn boundary (single-QA documents), wipe at the start of a
        # sentence instead -- trains reading-persistence across a sleep.
        if sentence_sleep_rate > 0 and sent_end_t is not None and space_start_t is not None:
            for i, ep in enumerate(chain):
                if len(pool_ids[ep]) < args.mid_sleep_min_len or rng.random() >= sentence_sleep_rate:
                    continue
                lo, hi = len(pool_ids[ep]) // 3, 2 * len(pool_ids[ep]) // 3
                seg, nxt = pool_ids[ep][lo:hi], pool_ids[ep][lo + 1 : hi + 1]
                m = torch.isin(seg, sent_end_t) & torch.isin(nxt, space_start_t)
                cand = (m.nonzero().flatten() + (offsets[i] + lo + 1)).tolist()
                if cand:
                    sleeps.append(rng.choice(cand))
                    n_sentence_sleeps += 1
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
        "dist_counts": dist_counts, "n_split": n_split, "n_split_qa": n_split_qa,
        "n_mid_sleeps": n_mid_sleeps, "n_sentence_sleeps": n_sentence_sleeps,
        "max_pending": max_pending, "n_sleep_chains": n_sleep_chains,
    }
    return dataset, stats


SUSPENDED = "USER->USER suspended (sleep between: a split-QA head awaiting its question)"
UU_SILENT = "USER->USER silent (no sleep between)"
AA = "ASST->ASST"


def validate(dataset, tokenizer, user_id: int, asst_id: int, n_samples: int = 2, ctx: int = 90) -> int:
    """Report structural invariants and decode a sample of each event, per the
    root CLAUDE.md rule. Returns the malformed-adjacency count.

    A well-formed stream alternates [USER] -> [ASSISTANT]. One exception:
    USER->USER across a sleep is a legal *suspension* -- a split-QA head's
    document turn suspended before its question, sleep-marked at the seam,
    resumed later behind a fresh user marker. USER->USER with no sleep means
    a splice left a turn unanswered, and ASST->ASST (an answer whose question
    is not in the stream) is malformed sleep or no sleep. Counts alone cannot
    catch this -- every count in every regen log was correct while the
    splices were malformed -- so this also prints the tokens.
    """
    counts = dict.fromkeys((SUSPENDED, UU_SILENT, AA), 0)
    samples: dict[str, list[str]] = {k: [] for k in counts}
    sleep_samples: list[str] = []
    ok = 0

    # The single special-token id is the only legal encoding of a role marker;
    # its multi-token BPE spelling means an episode was tokenized without the
    # special tokens registered (a stale pre-registration artifact).
    bpe_spellings = {
        tokenizer.convert_ids_to_tokens(mid): tokenizer(
            tokenizer.convert_ids_to_tokens(mid), add_special_tokens=False, split_special_tokens=True
        )["input_ids"]
        for mid in (user_id, asst_id)
    }
    n_bpe = dict.fromkeys(bpe_spellings, 0)
    n_marker = {user_id: 0, asst_id: 0}
    chains_no_marker = 0

    for ids, sleeps in zip(dataset["ids"], dataset["sleep_positions"]):
        n_u, n_a = int((ids == user_id).sum()), int((ids == asst_id).sum())
        n_marker[user_id] += n_u
        n_marker[asst_id] += n_a
        chains_no_marker += n_u + n_a == 0
        for marker, pat in bpe_spellings.items():
            if len(ids) < len(pat):
                continue
            hit = ids[: len(ids) - len(pat) + 1] == pat[0]
            for j in range(1, len(pat)):
                hit &= ids[j : len(ids) - len(pat) + 1 + j] == pat[j]
            n_bpe[marker] += int(hit.sum())
        marks = ((ids == user_id) | (ids == asst_id)).nonzero().flatten().tolist()
        sl = sleeps.tolist()
        for a, b in zip(marks, marks[1:]):
            pair = (int(ids[a]), int(ids[b]))
            if pair not in ((user_id, user_id), (asst_id, asst_id)):
                ok += 1
                continue
            slept = any(a < s <= b for s in sl)
            if pair == (user_id, user_id):
                kind = SUSPENDED if slept else UU_SILENT
            else:
                kind = AA
            counts[kind] += 1
            if len(samples[kind]) < n_samples:
                samples[kind].append(f"gap {b - a} tokens, sleep between: {slept}\n"
                                     f"    {tokenizer.decode(ids[max(0, b - ctx):b + ctx])!r}")
        for s in sl[:1]:
            if len(sleep_samples) < n_samples:
                sleep_samples.append(f"offset {s}\n    {tokenizer.decode(ids[max(0, s - ctx):s + ctx])!r}")

    bad = counts[UU_SILENT] + counts[AA]
    total = ok + sum(counts.values())
    n_chains = len(dataset["ids"])
    n_sleeping = sum(1 for s in dataset["sleep_positions"] if len(s))
    print(f"\nstructural validation ({total} role transitions):")
    print(f"  chains carrying sleeps: {n_sleeping}/{n_chains} "
          f"({100 * n_sleeping / max(n_chains, 1):.1f}%); the rest are plain concatenations")
    print(f"  marker ids: {n_marker[user_id]} user, {n_marker[asst_id]} assistant "
          f"(~{(n_marker[user_id] + n_marker[asst_id]) / max(n_chains, 1):.1f}/chain); "
          f"chains with no markers: {chains_no_marker}")
    print(f"  BPE-spelled markers (expected 0): "
          + ", ".join(f"{marker}: {n}" for marker, n in n_bpe.items()))
    for kind in counts:
        print(f"  {kind}: {counts[kind]}")
    if bad:
        print(f"  ** {bad} malformed ({100 * bad / total:.1f}%) -- every one of these is a turn "
              f"whose addressee is not in the stream; expected value is 0 **")
    for kind, exs in samples.items():
        for i, ex in enumerate(exs):
            print(f"\n  [{kind} sample {i + 1}] {ex}")
    for i, ex in enumerate(sleep_samples):
        print(f"\n  [sleep sample {i + 1}] {ex}")
    assert not any(n_bpe.values()), (
        f"BPE-spelled role markers found ({n_bpe}); the single special-token id is the only "
        f"legal marker encoding -- a source episode pool was tokenized without the special "
        f"tokens registered (regenerate it via prepare_data.py)"
    )
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sources", nargs="+", default=["data/train.pt", "data/train_memory.pt"],
                        help="Episode pools (prepare_data.py/merge_data.py outputs); every episode is used exactly once")
    parser.add_argument("--output", default="data/train_chains.pt")
    parser.add_argument("--min-budget", type=int, default=30_000, help="Per-chain token budget, log-uniform lower bound")
    parser.add_argument("--max-budget", type=int, default=130_000, help="Per-chain token budget, log-uniform upper bound")
    parser.add_argument("--min-wake", type=int, default=1, help="Minimum episodes per wake (between sleeps)")
    parser.add_argument("--max-wake", type=int, default=4, help="Maximum episodes per wake")
    parser.add_argument("--sleep-chain-rate", type=float, default=0.1,
                        help="Fraction of chains carrying the sleep apparatus at all (sleeps, splits, fact blocks). The rest are plain multi-episode concatenations with silent joins -- chains are deployment shape, retention pressure lives in the cram slices; 1.0 restores the fully-engineered dataset")
    parser.add_argument("--mid-sleep-rate", type=float, default=0.2,
                        help="Fraction of long episodes that get one mid-conversation sleep (the natural-continuation signal)")
    parser.add_argument("--split-episode-rate", type=float, default=0.0,
                        help="Fraction of eligible episodes split at a turn boundary (>= --split-min-part tokens on each side) with the tail resumed two episodes later behind a forced sleep -- trains cross-episode gist retention (interleaved continuation)")
    parser.add_argument("--split-min-part", type=int, default=256,
                        help="Minimum tokens on each side of a split-episode cut boundary")
    parser.add_argument("--split-qa-rate", type=float, default=None,
                        help="Split rate for single-QA episodes (exactly two turns). The cut lands at the question start recorded by prepare_data.py; the tail resumes behind a fresh [USER] marker with the question moved verbatim. Episodes without a recorded question never split. Default: --split-episode-rate")
    parser.add_argument("--split-gap-min", type=int, default=2,
                        help="Minimum episodes between a split head and its resumed tail")
    parser.add_argument("--split-gap-max", type=int, default=2,
                        help="Maximum episodes between a split head and its resumed tail (gap sampled per split)")
    parser.add_argument("--mid-sleep-min-len", type=int, default=4096,
                        help="Minimum episode length in tokens to be eligible for a mid-conversation sleep")
    parser.add_argument("--sentence-sleep-rate", type=float, default=0.0,
                        help="Fraction of long episodes (>= --mid-sleep-min-len) that get one sleep at a SENTENCE boundary in the middle third -- reaches inside long document turns where no turn boundary exists (reading-persistence signal)")
    parser.add_argument("--fact-rate", type=float, default=0.3, help="Fraction of episodes that host a fact block")
    parser.add_argument("--min-facts", type=int, default=4)
    parser.add_argument("--max-facts", type=int, default=64)
    parser.add_argument("--min-queries", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--revise-rate", type=float, default=0.12, help="Fraction of facts later revised to a new value")
    parser.add_argument("--cross-sleep-bias", type=float, default=0.0,
                        help="Probability of forcing a query to cross_sleep distance when that option exists (0.0 = uniform over available distances; only cross_sleep queries require the neural memory)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validate-samples", type=int, default=2,
                        help="Decoded samples printed per structural event kind")
    args = parser.parse_args()

    import models.mamba2_2_7b_memory as model_mod
    from models.common import build_tokenizer

    tokenizer = build_tokenizer(model_mod)
    labels = single_token_labels(tokenizer, LABEL_POOL, skip=LABEL_SKIP)

    pool_ids: list[torch.Tensor] = []
    pool_masks: list[torch.Tensor] = []
    pool_qoffs: list[int | None] = []
    for src in args.sources:
        data = torch.load(src, map_location="cpu", weights_only=True)
        pool_ids.extend(data["ids"])
        pool_masks.extend(data["masks"])
        qoffs = data.get("question_offsets") or [None] * len(data["ids"])
        pool_qoffs.extend(qoffs)
        print(f"  {src}: +{len(data['ids'])} episodes ({sum(q is not None for q in qoffs)} with question offsets)")

    sent_end_ids: set[int] = set()
    space_start_ids: set[int] = set()
    if args.sentence_sleep_rate > 0:
        # ponytail: vocab-string heuristic (byte-level BPE: Ġ=space, Ċ=newline);
        # upgrade to real sentence segmentation if boundary quality matters.
        for tok, tid in tokenizer.get_vocab().items():
            if tok.endswith((".", "!", "?")):
                sent_end_ids.add(tid)
            if tok.startswith(("Ġ", "Ċ")):
                space_start_ids.add(tid)
        print(f"sentence-boundary vocab scan: {len(sent_end_ids)} sentence-end ids, "
              f"{len(space_start_ids)} space-start ids")

    sep = tokenizer.encode(" ", add_special_tokens=False)
    assert len(sep) == 1, f'expected " " to be a single token, got {sep}'

    dataset, stats = build_chains(
        pool_ids, pool_masks,
        encode=lambda strings: tokenizer(strings, add_special_tokens=False)["input_ids"],
        labels=labels,
        user_open=model_mod.USER_OPEN, asst_open=model_mod.ASST_OPEN,
        user_id=tokenizer.convert_tokens_to_ids(model_mod.USER_OPEN),
        asst_id=tokenizer.convert_tokens_to_ids(model_mod.ASST_OPEN),
        args=args,
        sent_end_ids=sent_end_ids, space_start_ids=space_start_ids,
        pool_qoffs=pool_qoffs, sep_id=sep[0],
    )

    torch.save(dataset, args.output)
    total = sum(len(t) for t in dataset["ids"])
    n_sleep = sum(len(s) for s in dataset["sleep_positions"])
    n_recall = sum(int(r.sum()) for r in dataset["recall_masks"] if r is not None)
    print(
        f"wrote {args.output}: {len(dataset['ids'])} chains "
        f"({stats['n_sleep_chains']} planned sleeping, "
        f"{100 * stats['n_sleep_chains'] / max(len(dataset['ids']), 1):.1f}%), "
        f"{total / 1e6:.1f}M tokens, "
        f"{n_sleep} sleeps ({stats['n_mid_sleeps']} mid-conversation, "
        f"{stats['n_sentence_sleeps']} sentence-boundary, {stats['n_split']} split-tail, "
        f"{stats['n_blocks']} fact blocks, {stats['n_facts']} facts, "
        f"{stats['n_revised']} revisions, queries {stats['dist_counts']}), "
        f"{n_recall / 1e3:.1f}k recall-answer tokens"
    )
    print(f"max concurrent suspended episodes: {stats['max_pending']}")
    bad = validate(dataset, tokenizer,
                   tokenizer.convert_tokens_to_ids(model_mod.USER_OPEN),
                   tokenizer.convert_tokens_to_ids(model_mod.ASST_OPEN),
                   n_samples=args.validate_samples)
    if bad:
        # The artifact is already on disk -- inspect it, then regenerate.
        sys.exit(f"\n{args.output} has {bad} malformed role transitions; expected 0")

__all__ = [
    "sample_log_uniform",
    "build_chains",
    "SUSPENDED",
    "UU_SILENT",
    "AA",
    "validate",
    "main",
]


if __name__ == "__main__":
    main()
