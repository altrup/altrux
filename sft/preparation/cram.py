"""Wikipedia IMR cram blocks: the 35% retention slice of the next run
(notes/DISCUSSION-20260724-next-run-plan.md 1.3).

A cram block is one long training example of alternating turns:

  [USER] <fresh passage> <fresh passage> <re-shown sentence, entity blanked>
         <fresh passage>
  [ASSISTANT] <that sentence, entity restored>

Every token is dataset-authored -- the blank marker (`____`) is the only text
this script writes. The re-shown sentence sits at a *varied* position inside
the user turn (fresh passages follow it), so "answer the last thing you read"
is not a learnable policy, and the recall item doubles as the assistant turn,
so no user turn is ever left unanswered.

Three properties make the answer memory-dependent rather than retrievable:

- **Entity substitution.** The blanked entity is swapped for a same-type
  entity from another article, so the association exists only in this block's
  passages, never in pretrained weights.
- **Gap curriculum.** Each item's gap (tokens between its source passage and
  its cue) is drawn under a per-block ceiling that grows geometrically across
  the artifact -- the curriculum is encoded *data-side*, in emission order:
  block i carries ceiling_start * (ceiling_end/ceiling_start)^(i/n). Every
  item records its `gap`, `target_gap` and `ceiling`, so a consumer that
  shuffles can restore the curriculum by sorting on `ceiling`.
- **Credit on the entity span only.** recall_masks (train.py's --recall-weight
  multiplier) is True exactly on the entity tokens of the assistant turn. The
  rest of the completed sentence is copied from the visible cue and must earn
  no recall credit.

Solvability is *not* established here -- prepare_cram emits candidates and
filter_items.py decides which keep their recall credit (discarded items keep
their passages as interference; only the credit is dropped).

Held-out articles are reserved before item construction and written to a
separate artifact, so eval items can never come from a trained article.

  make prepare-cram
  uv run --no-sync python prepare_cram.py --articles 4000
"""

import argparse
import math
import random
import re
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

# CoNLL-03 groups from the default NER model. MISC (nationalities, events,
# adjectives) is excluded: a same-type swap inside it is rarely type-correct.
ENTITY_TYPES = ("PER", "ORG", "LOC")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CLEAN_SURFACE = re.compile(r"[A-Z][\w'’\-]*(?: [\w'’\-]+)*")
BLANK = "____"


def is_clean_surface(s: str) -> bool:
    """A usable entity surface: capitalized, no subword or punctuation junk.
    Keeps the substitution from producing text Wikipedia would never write."""
    return len(s) >= 3 and _CLEAN_SURFACE.fullmatch(s) is not None


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Char spans of each sentence.

    ponytail: punctuation-split heuristic (abbreviations split wrongly);
    upgrade to a real segmenter if cue quality ever measures badly.
    """
    spans, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def split_articles(titles, heldout_frac: float, seed: int) -> tuple[set[str], set[str]]:
    """Reserve whole articles for eval -- the Wikipedia analog of the
    held-out vocab slice. Deterministic in (titles, frac, seed)."""
    uniq = sorted(set(titles))
    n = round(heldout_frac * len(uniq))
    heldout = set(random.Random(seed).sample(uniq, n)) if n else set()
    return set(uniq) - heldout, heldout


def make_items(passage: dict, ents: list[dict], pool: dict[str, list[str]], rng,
               min_sentence_words: int = 8, max_items: int = 1) -> list[dict]:
    """A *group* of up to `max_items` cram items sharing one passage, or [].

    `ents` are NER spans over `passage["text"]`; `pool` maps entity type to
    candidate replacement surfaces (from other articles). Up to `max_items`
    entities are swapped for same-type candidates in a single pass, and each
    gets its own host sentence -- blanked it is that item's cue, intact it is
    the answer. Hosts are distinct and carry exactly one of the group's
    replacements, so no member's answer reveals another member's entity.

    One source passage answering several items is the interference-density
    knob: the block's tokens-per-item falls with the group size, and the
    tokens between any item's source and its cue are cue/answer turns rather
    than plain prose.
    """
    text = passage["text"]
    sents = [text[a:b] for a, b in split_sentences(text)]
    usable = [i for i, s in enumerate(sents)
              if len(s.split()) >= min_sentence_words and s[:1].isupper()
              and s.endswith((".", "!", "?"))]
    eligible = [e for e in ents if e["label"] in ENTITY_TYPES and is_clean_surface(e["text"])]
    rng.shuffle(eligible)

    def occurrences(s: str, word: str) -> int:
        # Word-boundary, not substring: "Principal" does not occur in
        # "Principality", and the substitution below will not touch it there.
        return len(re.findall(rf"(?<!\w){re.escape(word)}(?!\w)", s))

    chosen: list[tuple[str, str, str, int]] = []
    for ent in eligible:
        if len(chosen) >= max_items:
            break
        original, taken = ent["text"], {x for o, r, _, _ in chosen for x in (o, r)}
        if original in taken or any(occurrences(sents[h], original) for *_, h in chosen):
            continue
        hosts = [i for i in usable
                 if i not in {h for *_, h in chosen} and occurrences(sents[i], original) == 1
                 and not any(occurrences(sents[i], o) for o, *_ in chosen)]
        candidates = [s for s in pool.get(ent["label"], [])
                      if s != original and s not in text and s not in taken]
        if hosts and candidates:
            chosen.append((original, rng.choice(candidates), ent["label"], rng.choice(hosts)))
    if not chosen:
        return []

    # Word-boundary only, longest original first, one pass: a bare str.replace
    # turns "Principality" into "<swap>ity" when the entity is "Principal", and
    # substituting one entity at a time lets an earlier replacement be rewritten
    # by a later original.
    swaps = {o: r for o, r, _, _ in chosen}
    pattern = re.compile(r"(?<!\w)(" + "|".join(
        re.escape(o) for o in sorted(swaps, key=len, reverse=True)) + r")(?!\w)")

    def substitute(s: str) -> str:
        return pattern.sub(lambda m: swaps[m.group(1)], s)

    swapped = substitute(text)
    items = []
    for original, replacement, label, host in chosen:
        answer = substitute(sents[host])
        at = answer.index(replacement)
        items.append({
            "source": swapped,
            "cue": answer.replace(replacement, BLANK),
            "answer": answer,
            "span": (at, at + len(replacement)),
            "meta": {
                "article": passage["article"],
                "entity": replacement,
                "entity_type": label,
                "original_entity": original,
            },
        })
    return items


def _resolve_span(full: list[int], enc, answer: str, span: tuple[int, int], entity: str):
    """Token span of the entity inside the encoded answer, or None.

    Byte-level BPE merges a word with its leading space, so the char split at
    the entity's first letter often has no token boundary; retry with the
    space folded into the entity. Fails closed -- a misaligned span would put
    the x16 recall credit on the wrong tokens.
    """
    if answer[span[0]:span[1]] != entity:
        return None
    for start in (span[0], span[0] - 1):
        if start < 0 or (start != span[0] and answer[start] != " "):
            continue
        pre, ent = enc(answer[:start]), enc(answer[start:span[1]])
        if ent and full[:len(pre)] == pre and full[len(pre):len(pre) + len(ent)] == ent:
            return len(pre), len(pre) + len(ent)
    return None


def _sample_gap(rng: random.Random, lo: int, hi: int) -> int:
    lo = min(lo, hi)
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(max(hi, 1))))))


def _items_per_source(progress: float, args) -> int:
    """How many of a source's items a block at `progress` through the supply
    takes. Falls from `--items-per-source-start` to `--items-per-source-end`:
    low-ceiling blocks need their short gaps packed with cue/answer turns,
    high-ceiling ones get their interference from distance and volume."""
    a, b = args.items_per_source_start, args.items_per_source_end
    return max(1, int(round(a + (b - a) * progress)))


def build_blocks(groups: list[list[dict]], fillers: list[str], encode, *,
                 user_id: int, asst_id: int, sep_id: int, nl_id: int, args) -> tuple[dict, dict]:
    """The generator, IO-free: `encode` is a batched strings -> list[list[int]]
    callable, `args` the CLI namespace. Returns (dataset, stats).

    A group is the items sharing one source passage (see `make_items`); its
    passage is emitted once and every member takes its gap from that one
    span. Groups are consumed in order; the block's ceiling and its
    items-per-source are functions of how far through the supply it starts,
    so emission order *is* the curriculum.
    """
    strings: list[str] = []
    index: dict[str, int] = {}

    def add(s: str) -> int:
        if s not in index:
            index[s] = len(strings)
            strings.append(s)
        return index[s]

    items = [it for g in groups for it in g]
    for it in items:
        it["_i_src"], it["_i_cue"], it["_i_ans"] = add(it["source"]), add(it["cue"]), add(it["answer"])
        start, end = it["span"]
        for s in (start, start - 1):
            if s >= 0:
                add(it["answer"][:s])
                add(it["answer"][s:end])
    filler_idx = [add(f) for f in fillers]
    print(f"tokenizing {len(strings)} strings ({len(items)} items in {len(groups)} groups, "
          f"{len(fillers)} fillers) ...")
    enc = encode(strings) if strings else []

    def enc_of(s: str) -> list[int]:
        return enc[index[s]]

    resolved: list[list[dict]] = []
    n_resolved = 0
    for group in groups:
        keep = []
        for it in group:
            span = _resolve_span(enc[it["_i_ans"]], enc_of, it["answer"], it["span"],
                                 it["meta"]["entity"])
            if span is not None:
                it["_span"] = span
                keep.append(it)
        if keep:
            resolved.append(keep)
            n_resolved += len(keep)
    n_unresolved = len(items) - n_resolved

    rng = random.Random(args.seed)
    rng.shuffle(resolved)
    order = list(filler_idx)
    rng.shuffle(order)

    out_ids: list[torch.Tensor] = []
    out_masks: list[torch.Tensor] = []
    out_recall: list[torch.Tensor] = []
    out_sleeps: list[torch.Tensor] = []
    out_items: list[list[dict]] = []
    ceilings: list[int] = []
    ip = fp = 0
    n_forced = n_buried = n_leaked = 0

    while ip < len(resolved):
        progress = ip / len(resolved)
        ceiling = int(round(args.ceiling_start * (args.ceiling_end / args.ceiling_start) ** progress))
        per_source = _items_per_source(progress, args)
        budget = min(args.max_block_tokens, max(args.min_block_tokens, int(ceiling * args.block_gap_ratio)))
        ids: list[int] = []
        recall: list[bool] = []
        metas: list[dict] = []
        pending: list[dict] = []

        cap = getattr(args, "max_items_per_block", 0)
        introduced = 0

        def can_take_item() -> bool:
            return ip < len(resolved) and len(pending) < args.max_pending and not (cap and introduced >= cap)

        def can_emit() -> bool:
            return can_take_item() or bool(order)

        def take_group() -> list[dict]:
            nonlocal ip, introduced
            ip += 1
            group = resolved[ip - 1][:per_source]
            introduced += len(group)
            return group

        def next_unit():
            nonlocal fp
            can_item = can_take_item()
            can_filler = bool(order)
            if can_item and (not can_filler or rng.random() < args.item_rate):
                return "item", take_group()
            if can_filler:
                # The filler pool wraps rather than running out: exhausting it
                # would silently cut the artifact short of its items. The cycle
                # count in the summary says whether to stream more articles.
                fp += 1
                return "filler", enc[order[(fp - 1) % len(order)]]
            if can_item:
                return "item", take_group()
            return None

        while len(ids) < budget and (ip < len(resolved) or pending):
            turn = [user_id, sep_id]
            cue_item = None
            tail_left = 0
            while True:
                pos = len(ids) + len(turn)
                if cue_item is None:
                    due = next((it for it in pending if pos - it["source_end"] >= it["target_gap"]), None)
                    if due is None and pending and (pos >= budget or not can_emit()):
                        # Closing the turn early: probe the longest-waiting item,
                        # but never below the gap floor -- a cue right behind its
                        # source is SSM-trivial and would train against memory.
                        best = max(pending, key=lambda it: pos - it["source_end"])
                        if pos - best["source_end"] >= args.gap_min:
                            due = best
                            n_forced += 1
                    if due is not None:
                        pending.remove(due)
                        sep = [nl_id] if len(turn) > 2 else []
                        cue_ids = enc[due["_i_cue"]]
                        due["cue_start"] = pos + len(sep)
                        due["cue_end"] = due["cue_start"] + len(cue_ids)
                        due["gap"] = due["cue_start"] - due["source_end"]
                        turn += sep + cue_ids
                        cue_item = due
                        tail_left = rng.randint(0, args.max_tail_units)
                        continue
                elif tail_left <= 0 or any(pos - it["source_end"] >= it["target_gap"] for it in pending):
                    # Cut the tail short when the next item comes due, so a
                    # realized gap overshoots its target by at most one passage
                    # plus the answer turn.
                    break
                unit = next_unit()
                if unit is None:
                    break
                kind, payload = unit
                sep = [nl_id] if len(turn) > 2 else []
                if kind == "item":
                    group = payload
                    unit_ids = enc[group[0]["_i_src"]]
                    start = len(ids) + len(turn) + len(sep)
                    for it in group:
                        it["source_start"] = start
                        it["source_end"] = start + len(unit_ids)
                        it["target_gap"] = _sample_gap(rng, args.gap_min, ceiling)
                        it["ceiling"] = ceiling
                    pending += group
                else:
                    unit_ids = payload
                turn += sep + unit_ids
                if cue_item is not None:
                    tail_left -= 1

            if cue_item is None:
                break  # supply exhausted mid-turn: drop the unanswered turn
            answer_ids = enc[cue_item["_i_ans"]]
            answer_start = len(ids) + len(turn)
            turn += [asst_id, sep_id] + answer_ids
            s, e = cue_item["_span"]
            span_start = answer_start + 2 + s
            ids += turn
            recall += [False] * (len(ids) - len(recall))
            for i in range(span_start, answer_start + 2 + e):
                recall[i] = True
            metas.append({
                "gap": cue_item["gap"], "target_gap": cue_item["target_gap"], "ceiling": ceiling,
                "source_start": cue_item["source_start"], "source_end": cue_item["source_end"],
                "cue_start": cue_item["cue_start"], "cue_end": cue_item["cue_end"],
                "answer_start": answer_start, "span_start": span_start,
                "span_end": answer_start + 2 + e,
                "credit_text": cue_item["answer"][cue_item["span"][0]:cue_item["span"][1]],
                **cue_item["meta"],
            })
            if cap and len(metas) >= cap:
                break

        if not metas:
            break
        # An entity that also occurs somewhere else in the block (its
        # replacement surface is a real entity elsewhere in the corpus) is
        # readable without memory -- keep the passages, drop the credit.
        ids_t = torch.tensor(ids, dtype=torch.long)
        recall_t = torch.tensor(recall, dtype=torch.bool)
        kept = []
        for it in metas:
            span = ids_t[it["span_start"]:it["span_end"]]
            stray = [at for at in _find(ids_t, span)
                     if at != it["span_start"] and not (it["source_start"] <= at < it["source_end"])]
            if stray and not getattr(args, "allow_repeated_credit", False):
                recall_t[it["span_start"]:it["span_end"]] = False
                n_leaked += 1
            else:
                kept.append(it)
                n_buried += it["cue_end"] < it["answer_start"]
        metas = kept
        if not metas:
            continue
        out_ids.append(ids_t)
        out_masks.append(torch.ones(len(ids), dtype=torch.bool))
        out_recall.append(recall_t)
        out_sleeps.append(torch.zeros(0, dtype=torch.long))
        out_items.append(metas)
        ceilings.append(ceiling)
        print(f"\r  built {len(out_ids)} blocks, {sum(len(m) for m in out_items)} items, "
              f"ceiling {ceiling}", end="", flush=True)
    print()

    dataset = {
        "ids": out_ids, "masks": out_masks, "recall_masks": out_recall,
        "sleep_positions": out_sleeps, "items": out_items,
        "curriculum": {
            "ceilings": ceilings, "gap_min": args.gap_min,
            "ceiling_start": args.ceiling_start, "ceiling_end": args.ceiling_end,
            "order": "emission order is the curriculum; sort blocks by ceiling to restore it",
        },
    }
    stats = {
        "n_blocks": len(out_ids), "n_items": sum(len(m) for m in out_items),
        "n_span_unresolved": n_unresolved, "n_forced_cues": n_forced,
        "n_buried_cues": n_buried, "n_leaked_dropped": n_leaked,
        "n_tokens": sum(len(t) for t in out_ids),
        "filler_cycles": fp / len(order) if order else 0.0,
    }
    return dataset, stats


def find_subsequence(ids: torch.Tensor, pat: torch.Tensor) -> list[int]:
    n, m = len(ids), len(pat)
    if m == 0 or n < m:
        return []
    hit = torch.ones(n - m + 1, dtype=torch.bool)
    for j in range(m):
        hit &= ids[j:n - m + 1 + j] == pat[j]
    return hit.nonzero().flatten().tolist()


_find = find_subsequence


def validate_blocks(dataset, tokenizer, *, user_id: int, asst_id: int,
                    n_samples: int = 2, ctx: int = 220, unique_credit: bool = True) -> dict:
    """Structural invariants (every count below is correct only at zero) plus
    decoded text around one instance of each structural event, per the root
    CLAUDE.md rule. Counts confirm the generator did what it was told; only
    the decoded text confirms what it was told was right.
    """
    report = dict.fromkeys(
        ("malformed_adjacency", "span_text_mismatch", "credit_visible_before_cue",
         "stray_entity_occurrences", "credit_outside_recorded_spans"), 0)
    gaps: list[int] = []
    samples: list[str] = []
    density: list[tuple[int, int]] = []

    for ids, recall, block in zip(dataset["ids"], dataset["recall_masks"], dataset["items"]):
        cues = sorted(it["cue_start"] for it in block)
        density += [(it["gap"], sum(it["source_end"] <= c < it["cue_start"] for c in cues))
                    for it in block]
        marks = ((ids == user_id) | (ids == asst_id)).nonzero().flatten().tolist()
        for a, b in zip(marks, marks[1:]):
            report["malformed_adjacency"] += int(ids[a]) == int(ids[b])
        credited = torch.zeros(len(ids), dtype=torch.bool)
        for it in block:
            credited[it["span_start"]:it["span_end"]] = True
            gaps.append(it["gap"])
            span = ids[it["span_start"]:it["span_end"]]
            report["span_text_mismatch"] += tokenizer.decode(span).strip() != it["credit_text"].strip()
            for at in _find(ids, span):
                if it["source_start"] <= at < it["source_end"] or at == it["span_start"]:
                    continue
                report["stray_entity_occurrences"] += 1
                report["credit_visible_before_cue"] += it["source_end"] <= at < it["cue_start"]
            if len(samples) < n_samples:
                samples.append(
                    f"gap {it['gap']} (target {it['target_gap']}, ceiling {it['ceiling']}), "
                    f"entity {it['entity']!r} ({it['entity_type']}), article {it['article']!r}\n"
                    f"    [source]  ...{tokenizer.decode(ids[it['source_start']:it['source_end']])[:ctx]}...\n"
                    f"    [cue]     ...{tokenizer.decode(ids[max(0, it['cue_start'] - ctx):it['cue_end']])}\n"
                    f"    [answer]  {tokenizer.decode(ids[it['answer_start']:it['span_start']])}"
                    f"<<{tokenizer.decode(ids[it['span_start']:it['span_end']])}>>"
                    f"{tokenizer.decode(ids[it['span_end']:it['span_end'] + 60])}\n"
                    f"    (<<>> marks the recall-credited span; everything else is uncredited)"
                )
        report["credit_outside_recorded_spans"] += int((recall & ~credited).sum())

    gaps_sorted = sorted(gaps)
    informational = set() if unique_credit else {"stray_entity_occurrences", "credit_visible_before_cue"}
    print(f"\nstructural validation ({len(gaps)} items in {len(dataset['ids'])} blocks):")
    for k, v in report.items():
        note = "informational: closed-vocabulary answers recur" if k in informational else "expected 0"
        print(f"  {k} ({note}): {v}")
    if gaps_sorted:
        print(f"  gap tokens: min {gaps_sorted[0]}, median {gaps_sorted[len(gaps_sorted) // 2]}, "
              f"max {gaps_sorted[-1]}; {sum(g < 192 for g in gaps_sorted)} below the 192-token "
              f"SSM-interference floor (RESEARCH-20260724-local-diagnostics.md 1)")
        n_tokens = sum(len(t) for t in dataset["ids"])
        short = sorted(n for g, n in density if g <= 512)
        print(f"  interference density: {n_tokens / len(gaps):.0f} tokens per item; "
              f"median {sorted(n for _, n in density)[len(density) // 2]} cue/answer turns inside a gap"
              + (f", {short[len(short) // 2]} inside the {len(short)} gaps at or below 512 tokens"
                 if short else ", no gaps at or below 512 tokens"))
    for i, s in enumerate(samples):
        print(f"\n  [item sample {i + 1}] {s}")
    return report


def load_wikipedia_passages(args) -> list[dict]:
    """Stream `--wiki-dataset` and cut each article into paragraph-sized
    passages. Streaming keeps this to the first shards instead of the full
    ~20 GB parquet download."""
    from datasets import load_dataset

    ds = load_dataset(args.wiki_dataset, args.wiki_config, split="train", streaming=True)
    passages: list[dict] = []
    seen = 0
    for row in ds:
        if seen >= args.articles:
            break
        seen += 1
        kept = 0
        for para in row["text"].split("\n\n"):
            para = para.strip()
            if kept >= args.passages_per_article:
                break
            if args.min_words <= len(para.split()) <= args.max_words:
                passages.append({"article": row["title"], "text": para})
                kept += 1
        if seen % 200 == 0:
            print(f"\r  {seen}/{args.articles} articles, {len(passages)} passages", end="", flush=True)
    print(f"\r  {seen}/{args.articles} articles, {len(passages)} passages")
    return passages


def annotate(texts: list[str], args) -> list[list[dict]]:
    """NER over every passage.

    Deviation from the plan's spaCy choice: this venv is Python 3.14, which
    spaCy has no wheels for (a source build of thinc/blis, plus the
    documented risk that any uv install clobbers the ROCm torch build).
    transformers is already a dependency and its CoNLL-03 token classifier
    gives the PER/ORG/LOC types the same-type swap needs.
    """
    import torch as _torch
    from transformers import pipeline

    device = args.ner_device if args.ner_device is not None else (0 if _torch.cuda.is_available() else -1)
    print(f"NER: {args.ner_model} on device {device} over {len(texts)} passages ...")
    ner = pipeline("token-classification", model=args.ner_model,
                   aggregation_strategy="simple", device=device)
    out: list[list[dict]] = []
    for i in range(0, len(texts), args.ner_batch):
        batch = texts[i:i + args.ner_batch]
        for text, res in zip(batch, ner(batch)):
            # Sliced from the source text, not taken from e["word"] -- the
            # aggregated pipeline reports subword pieces there ("##orra").
            out.append([{"text": text[int(e["start"]):int(e["end"])], "label": e["entity_group"],
                         "start": int(e["start"]), "end": int(e["end"])} for e in res])
        print(f"\r  {len(out)}/{len(texts)} passages", end="", flush=True)
    print()
    return out


def build_items(passages: list[dict], ents: list[list[dict]], seed: int,
                min_sentence_words: int, max_items_per_source: int = 1,
                ) -> tuple[list[list[dict]], list[str]]:
    """Item groups (one per usable passage) plus the passages that yielded
    none (kept as filler/interference).

    Replacement surfaces are drawn without replacement so no two items in the
    artifact share a fabricated entity.
    """
    rng = random.Random(seed)
    pool: dict[str, list[str]] = {}
    for es in ents:
        for e in es:
            if e["label"] in ENTITY_TYPES and is_clean_surface(e["text"]):
                pool.setdefault(e["label"], []).append(e["text"])
    for label in pool:
        pool[label] = sorted(set(pool[label]))
        rng.shuffle(pool[label])

    groups, leftover, n_items = [], [], 0
    for i, (p, es) in enumerate(zip(passages, ents)):
        group = make_items(p, es, pool, rng, min_sentence_words=min_sentence_words,
                           max_items=max_items_per_source)
        if not group:
            leftover.append(p["text"])
        else:
            groups.append(group)
            n_items += len(group)
            for it in group:
                used = it["meta"]["entity"]
                pool[it["meta"]["entity_type"]] = [s for s in pool[it["meta"]["entity_type"]] if s != used]
        if i % 200 == 0:
            print(f"\r  {i + 1}/{len(passages)} passages, {n_items} items", end="", flush=True)
    print(f"\r  {len(passages)}/{len(passages)} passages, {n_items} items in {len(groups)} groups "
          f"({n_items / max(len(groups), 1):.2f} per source)")
    return groups, leftover


def add_block_args(parser) -> None:
    """Block-assembly and curriculum flags, shared with prepare_needles.py."""
    parser.add_argument("--gap-min", type=int, default=192,
                        help="Floor on the per-item gap (tokens between a source passage and its cue). "
                             "192 is where dense interference kills plain-backbone recall "
                             "(notes/RESEARCH-20260724-local-diagnostics.md 1)")
    parser.add_argument("--ceiling-start", type=int, default=448,
                        help="Gap ceiling for the first block -- inside the 512-token BPTT window, so early "
                             "recall loss can credit the write end to end")
    parser.add_argument("--ceiling-end", type=int, default=8192, help="Gap ceiling for the last block")
    parser.add_argument("--block-gap-ratio", type=float, default=4.0,
                        help="Block token budget as a multiple of its gap ceiling")
    parser.add_argument("--min-block-tokens", type=int, default=2048)
    parser.add_argument("--max-block-tokens", type=int, default=32768)
    parser.add_argument("--max-tail-units", type=int, default=3,
                        help="Upper bound on passages emitted AFTER the cue in the same user turn -- what "
                             "keeps the cue off the turn's tail and defeats answer-the-last-thing")
    parser.add_argument("--item-rate", type=float, default=1.0,
                        help="Probability a fresh unit is a probed item source rather than plain filler")
    parser.add_argument("--max-pending", type=int, default=64,
                        help="Cap on items awaiting their cue at once")
    parser.add_argument("--items-per-source-start", type=int, default=3,
                        help="Items a source passage answers in the FIRST (lowest-ceiling) block -- the "
                             "interference-density knob. A turn costs one passage plus one cue/answer "
                             "pair, so sharing a passage across k items cuts the block's tokens per item "
                             "and packs that many more fabricated entities into every token of gap. 3 is "
                             "where Wikipedia's yield flattens: 1.69 items/passage measured, 251 -> 191 "
                             "tokens per item and 1.7 -> 2.25 cue/answer turns inside a 512-token gap, "
                             "for +47%% items off the same articles; 4 buys 2%% more density for 84%% more "
                             "credit lost to entity leaks")
    parser.add_argument("--items-per-source-end", type=int, default=1,
                        help="Items a source passage answers in the LAST (highest-ceiling) block, "
                             "interpolated linearly from --items-per-source-start across the artifact. "
                             "Long-gap items get their interference from distance and volume, so density "
                             "relaxes and the item supply goes further")
    parser.add_argument("--max-items-per-block", type=int, default=0,
                        help="Close the block after this many answered items (0 = fill the budget). Needles "
                             "use 1: bAbI stories share a six-name vocabulary, so a second story in the same "
                             "block re-answers the first one's question and the credited target goes stale")
    parser.add_argument("--allow-repeated-credit", action="store_true",
                        help="Keep items whose credited answer also occurs elsewhere in their block. Off for "
                             "cram (a fabricated entity that recurs really is readable without memory); on "
                             "for needles, whose closed-vocabulary answers recur by construction while the "
                             "binding the question asks about does not")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--validate-samples", type=int, default=2)


def emit(dataset: dict, stats: dict, path: str, tokenizer, user_id: int, asst_id: int,
         slice_name: str, heldout: list[str], args) -> None:
    dataset["slice"] = slice_name
    dataset["heldout_articles"] = sorted(heldout)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out)
    n_recall = sum(int(r.sum()) for r in dataset["recall_masks"])
    print(f"\nwrote {out}: {stats['n_blocks']} blocks, {stats['n_tokens'] / 1e6:.2f}M tokens, "
          f"{stats['n_items']} items ({n_recall} recall-credited tokens, "
          f"{100 * n_recall / max(stats['n_tokens'], 1):.2f}% of tokens), "
          f"{stats['n_span_unresolved']} items dropped on span misalignment, "
          f"{stats['n_leaked_dropped']} on an entity visible elsewhere in the block, "
          f"{stats['n_forced_cues']} cues forced early, "
          f"{stats['n_buried_cues']}/{stats['n_items']} cues followed by more passages in their turn, "
          f"filler pool used {stats['filler_cycles']:.1f}x (stream more articles if that is far above 1)")
    unique_credit = not getattr(args, "allow_repeated_credit", False)
    report = validate_blocks(dataset, tokenizer, user_id=user_id, asst_id=asst_id,
                             n_samples=args.validate_samples, unique_credit=unique_credit)
    if not unique_credit:
        for k in ("stray_entity_occurrences", "credit_visible_before_cue"):
            report.pop(k)
    if any(report.values()):
        print(f"  ** {sum(report.values())} structural violations -- expected 0 **")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default="data/train_cram.pt")
    parser.add_argument("--heldout-output", default="data/eval_cram.pt",
                        help="Blocks built from the held-out articles, same schema -- eval items must never "
                             "come from a trained article")
    parser.add_argument("--articles", type=int, default=8000, help="Wikipedia articles to stream")
    parser.add_argument("--wiki-dataset", default="wikimedia/wikipedia")
    parser.add_argument("--wiki-config", default="20231101.en")
    parser.add_argument("--passages-per-article", type=int, default=3)
    parser.add_argument("--min-words", type=int, default=45, help="Shortest usable passage")
    parser.add_argument("--max-words", type=int, default=140, help="Longest usable passage")
    parser.add_argument("--min-sentence-words", type=int, default=8, help="Shortest usable cue sentence")
    parser.add_argument("--heldout-frac", type=float, default=0.05, help="Fraction of ARTICLES reserved for eval")
    parser.add_argument("--ner-model", default="dslim/bert-base-NER")
    parser.add_argument("--ner-batch", type=int, default=64)
    parser.add_argument("--ner-device", type=int, default=None, help="Default: GPU if one is visible, else CPU")
    add_block_args(parser)
    args = parser.parse_args()

    import importlib
    import os

    from models.common import build_tokenizer

    model_mod = importlib.import_module(f"models.{os.getenv('MODEL_NAME', 'mamba2_780m')}")
    tokenizer = build_tokenizer(model_mod)
    user_id = tokenizer.convert_tokens_to_ids(model_mod.USER_OPEN)
    asst_id = tokenizer.convert_tokens_to_ids(model_mod.ASST_OPEN)
    sep = tokenizer.encode(" ", add_special_tokens=False)
    nl = tokenizer.encode("\n", add_special_tokens=False)
    assert len(sep) == 1 and len(nl) == 1, f'expected " " and "\\n" to be single tokens, got {sep} {nl}'

    passages = load_wikipedia_passages(args)
    train_titles, heldout_titles = split_articles([p["article"] for p in passages],
                                                  args.heldout_frac, args.seed)
    ents = annotate([p["text"] for p in passages], args)

    def encode(strings: list[str]) -> list[list[int]]:
        return tokenizer(strings, add_special_tokens=False)["input_ids"]

    for name, titles, path in (("cram", train_titles, args.output),
                               ("cram-heldout", heldout_titles, args.heldout_output)):
        sel = [i for i, p in enumerate(passages) if p["article"] in titles]
        print(f"\n== {name}: {len(sel)} passages from {len(titles)} articles ==")
        groups, leftover = build_items([passages[i] for i in sel], [ents[i] for i in sel],
                                       args.seed, args.min_sentence_words,
                                       max(args.items_per_source_start, args.items_per_source_end))
        dataset, stats = build_blocks(groups, leftover, encode, user_id=user_id, asst_id=asst_id,
                                      sep_id=sep[0], nl_id=nl[0], args=args)
        emit(dataset, stats, path, tokenizer, user_id, asst_id, name, sorted(heldout_titles), args)

__all__ = [
    "add_block_args",
    "build_blocks",
    "emit",
    "find_subsequence",
    "load_wikipedia_passages",
    "make_items",
    "split_articles",
    "split_sentences",
    "validate_blocks",
    "_find",
    "main",
]


if __name__ == "__main__":
    main()
