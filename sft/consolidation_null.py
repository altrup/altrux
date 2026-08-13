"""Transcript-consolidation null: can a brief LoRA distillation pass install
facts that were held losslessly in context into the *weights*, such that the
model generates them later from a fresh state with no context at all?

This is the field-default consolidation baseline (SEAL / Cartridges /
2605.26099 all source their fine-tuning data from context, not from a memory
module -- see notes/DISCUSSION-20260725-cl-sleep-analysis-and-filter-testc.md
sec 4 and notes/DISCUSSION-20260730-ssm-consume-on-read-and-m-necessity.md
sec 4). It gates everything downstream: if lossless-transcript distillation
cannot install recallable facts, neither M-replay nor SSM-replay can, and
CL-by-consolidation needs a rethink before any training budget is spent. If
it works, it is the harness and the bar for the M-replay arm.

Shape of the run:

  wake      -- a synthetic transcript states N entity->code facts, separated
               by digit-free filler turns, in the training chat format.
  pre       -- three probes before any distillation: the in-context positive
               control (transcript in state, must be high or the harness is
               broken), the fresh-state floor (must be ~0), and the
               teacher-forced log-prob of each code from a fresh state.
  sleep     -- LoRA distillation. Teacher = the frozen model over
               [transcript || transcript-replay], logits taken on the replay
               half (so the teacher answers with the facts in context).
               Student = the LoRA model over the replay half alone from a
               fresh state. Loss = KL(teacher || student) at --kl-temp. The
               teacher's logits are cached once, before the first optimizer
               step, while the adapters are still identity (lora_B is
               zero-init, and a loaded --checkpoint is likewise frozen at
               that moment) -- so one model instance serves as both.
  post      -- fresh state, no context, LoRA attached: greedy exact match,
               pass@k, a three-rung cue ladder (free recall -> category hint
               -> first-digit hint), and the log-prob delta vs the floor.

Verdict is three-way against pre-registered thresholds (PASS_MATCH_RATE,
UNDERPOWERED_DELTA_NATS): PASS / FAIL-UNDERPOWERED (the distribution moved
but generation does not produce the fact) / FAIL-DEAD.

Box tool: this runs on rented CUDA hardware, never the local ROCm box -- it
holds a full-vocab teacher-logit cache for the whole replay and trains a
LoRA on top of it. Only the pure transcript/grading logic is CPU-testable
(tests/test_consolidation_null.py), which is why every torch and models.*
import lives inside the functions that need it.

Usage (from sft/, env vars as in the Makefile):
    make consolidation-null ARGS="--n-facts 40 --distill-steps 200"
    make consolidation-null ARGS="--checkpoint ../models/mamba2_780m/checkpoints/epoch-1/step-500"
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from experiments.facts import (
    CODE_DIGITS,
    ENTITY_POOL,
    FILLER_SENTENCES,
    Fact,
    Turn,
    build_facts,
    build_turns,
    contains_code,
    cue_rungs,
    digits,
    exact_match,
    extract_answer,
    fact_turns,
    first_success,
    normalize,
    pass_at_k,
    render_turns,
    role_adjacency_violations,
)
from experiments.inference import generate, kl_loss, replay_step, run_chunks, target_logprob
from progress import fmt_duration, ts

if TYPE_CHECKING:
    import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Pre-registered before the first run: a greedy exact-match rate at or above
# PASS_MATCH_RATE is a PASS; below it, a mean log-prob gain of at least
# UNDERPOWERED_DELTA_NATS per code token separates "moved, but not enough to
# generate" from "nothing happened".
PASS_MATCH_RATE = 0.30
UNDERPOWERED_DELTA_NATS = 1.0

GEN_TOKENS = 16


def verdict(match_rate: float, mean_delta_nats: float) -> str:
    if match_rate >= PASS_MATCH_RATE:
        return "PASS"
    if mean_delta_nats >= UNDERPOWERED_DELTA_NATS:
        return "FAIL-UNDERPOWERED"
    return "FAIL-DEAD"


def report_transcript(text: str, decoded: str, facts: Sequence[Fact], turns: Sequence[Turn], n_tokens: int) -> None:
    """Structural invariants whose correct value is zero, plus decoded text
    around one fact and one filler join -- counts alone have shipped malformed
    transcripts before (root CLAUDE.md)."""
    misstated = sum(1 for f in facts if decoded.count(f"is {f.code}.") != 1)
    missing_entities = sum(1 for f in facts if decoded.count(f.entity) != 2)
    stray_digits = sum(1 for s in FILLER_SENTENCES if any(c.isdigit() for c in s))
    print(f"[{ts()}] transcript: {len(facts)} facts, {len(turns)} turns, {n_tokens} tokens")
    print(f"[{ts()}]   role-adjacency violations : {role_adjacency_violations(turns)}  (must be 0)")
    print(f"[{ts()}]   facts not stated exactly once: {misstated}  (must be 0)")
    print(f"[{ts()}]   entities not mentioned twice : {missing_entities}  (must be 0)")
    print(f"[{ts()}]   filler sentences with digits : {stray_digits}  (must be 0)")
    print(f"[{ts()}]   round-trip decode == source  : {decoded == text}")

    at = decoded.find(facts[0].code)
    if at < 0:
        print(f"[{ts()}]   sample around fact 0: NOT FOUND in the decoded transcript -- tokenization broke the code")
    else:
        print(f"[{ts()}]   sample around fact 0:\n    ...{decoded[max(0, at - 200) : at + 200]!r}...")
    join = decoded.find(facts[1].entity) if len(facts) > 1 else -1
    if join > 0:
        print(f"[{ts()}]   sample around the filler->fact join:\n    ...{decoded[max(0, join - 300) : join + 100]!r}...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=None, help="Checkpoint dir with trainable.pt to distil on top of (default: the base model)")
    parser.add_argument("--n-facts", type=int, default=40, help="Facts in the wake transcript (default: %(default)s)")
    parser.add_argument("--filler-tokens", type=int, default=200, help="Filler tokens between consecutive facts (default: %(default)s)")
    parser.add_argument("--distill-steps", type=int, default=200, help="Optimizer steps; one step = one replay chunk (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default: %(default)s)")
    parser.add_argument("--kl-temp", type=float, default=1.0, help="Distillation temperature (default: %(default)s)")
    parser.add_argument("--chunk-len", type=int, default=None, help="Tokens per forward chunk (default: the model's DEFAULT_CHUNK_LEN)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--pass-k", type=int, default=10, help="Samples per fact for pass@k (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for pass@k (default: %(default)s)")
    parser.add_argument("--gen-tokens", type=int, default=GEN_TOKENS, help="Tokens generated per probe (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="logs/consolidation_null.jsonl", help="Per-fact results jsonl (default: %(default)s)")
    parser.add_argument(
        "--fresh-state-replay",
        action="store_true",
        help="reset the student's state before every replay chunk, not just at pass boundaries",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="disable the neural-memory injection, distilling into the backbone alone",
    )
    args = parser.parse_args()

    import importlib
    import os

    import torch
    from dotenv import load_dotenv

    load_dotenv()
    model_name = os.getenv("MODEL_NAME", "mamba2_780m")
    model_mod = importlib.import_module(f"models.{model_name}")
    train_hooks = importlib.import_module(f"models.{model_name}.train_hooks")
    from models.common import build_tokenizer, set_memory_injection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_len = args.chunk_len or getattr(train_hooks, "DEFAULT_CHUNK_LEN", 48)
    print(f"[{ts()}] model {model_name} on {device}, chunk_len {chunk_len}, seed {args.seed}")

    rank, alpha = args.lora_rank, args.lora_alpha
    if args.checkpoint:
        cfg_path = Path(args.checkpoint) / "lora_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            rank, alpha = cfg["rank"], cfg["alpha"]
    model, trainable = train_hooks.setup_training(device, rank, alpha, 0.0)
    if args.checkpoint:
        from train import load_checkpoint

        load_checkpoint(model, Path(args.checkpoint))
    model.eval()
    has_memory = set_memory_injection(model, not args.no_memory)
    print(f"[{ts()}] memory injection: {'absent' if not has_memory else ('off' if args.no_memory else 'on')}")
    tokenizer = build_tokenizer(model_mod)
    user_open, asst_open = model_mod.USER_OPEN, model_mod.ASST_OPEN
    stops = (".", "\n", user_open, asst_open)

    def encode(text: str) -> torch.Tensor:
        return torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]], dtype=torch.long, device=device)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    facts = build_facts(args.n_facts, rng)
    def token_len(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False)["input_ids"])

    turns = build_turns(facts, args.filler_tokens, token_len, rng)
    text = render_turns(turns, user_open, asst_open)
    transcript = encode(text)
    report_transcript(text, tokenizer.decode(transcript[0].cpu()), facts, turns, transcript.shape[1])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.open("w")

    def emit(record: dict[str, object]) -> None:
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()

    print(f"\n[{ts()}] === pre-distillation probes ===")
    _, primed = run_chunks(model, transcript, None, chunk_len, "prime", keep_logits=False)

    pre: list[dict[str, object]] = []
    n_ctx = n_floor = 0
    for i, fact in enumerate(facts):
        prompt = encode(cue_rungs(fact, user_open, asst_open)[0][0])
        target = encode(" " + fact.code)
        in_ctx = tokenizer.decode(generate(model, prompt, copy.deepcopy(primed), args.gen_tokens, 0.0)[0].cpu())
        fresh = tokenizer.decode(generate(model, prompt, None, args.gen_tokens, 0.0)[0].cpu())
        lp = target_logprob(model, prompt, target, None)
        n_ctx += exact_match(in_ctx, fact.code, stops)
        n_floor += exact_match(fresh, fact.code, stops)
        pre.append({"in_context": in_ctx, "fresh": fresh, "logprob": lp})
        record = {
            "phase": "pre", "fact": fact.entity, "category": fact.category, "code": fact.code,
            "in_context": in_ctx, "in_context_match": exact_match(in_ctx, fact.code, stops),
            "fresh": fresh, "fresh_match": exact_match(fresh, fact.code, stops), "logprob": lp,
        }
        emit(record)
        print(
            f"\r[{ts()}]  pre {i + 1}/{len(facts)} {fact.entity:<11} "
            f"in-context {n_ctx / (i + 1):.2f}  fresh-floor {n_floor / (i + 1):.2f}",
            end="", flush=True,
        )
    print()
    ctx_rate, floor_rate = n_ctx / len(facts), n_floor / len(facts)
    pre_logprob = sum(float(p["logprob"]) for p in pre) / len(pre)
    print(f"[{ts()}] IN-CONTEXT POSITIVE CONTROL: {ctx_rate:.3f} exact match "
          f"({'ok' if ctx_rate >= 0.8 else 'LOW -- the harness, not the hypothesis, is what this measures'})")
    print(f"[{ts()}] fresh-state floor: {floor_rate:.3f} exact match, mean code log-prob {pre_logprob:+.4f}")
    del primed
    torch.cuda.empty_cache()

    # Teacher logits are cached once, here, while the adapters are still
    # identity (lora_B is zero-init and a loaded --checkpoint is frozen at
    # this moment), so the same instance serves as frozen teacher and student.
    print(f"\n[{ts()}] === distillation ===")
    _, ctx_state = run_chunks(model, transcript, None, chunk_len, "teacher context", keep_logits=False)
    teacher_logits, _ = run_chunks(model, transcript, ctx_state, chunk_len, "teacher replay")
    del ctx_state
    torch.cuda.empty_cache()
    print(f"[{ts()}] teacher logit cache: {tuple(teacher_logits.shape)} on cpu "
          f"({teacher_logits.element_size() * teacher_logits.nelement() / 2**30:.2f} GiB)")

    # One step = one replay chunk. State is carried (detached) across chunks
    # within a pass and reset to None at each pass boundary, so the student
    # always sees the replay from a fresh state -- truncated BPTT, the same
    # shape train.py uses.
    model.train()
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    n_chunks = (transcript.shape[1] + chunk_len - 1) // chunk_len
    state = None
    started = time.time()
    for step in range(args.distill_steps):
        c, reset = replay_step(step, n_chunks, args.fresh_state_replay)
        if reset:
            state = None
        lo, hi = c * chunk_len, min((c + 1) * chunk_len, transcript.shape[1])
        logits, state = model(transcript[:, lo:hi], state=state)
        state = state.detach()
        loss = kl_loss(teacher_logits[:, lo:hi].to(device), logits, args.kl_temp)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        rate = (step + 1) / (time.time() - started)
        print(
            f"\r[{ts()}]  distill step {step + 1}/{args.distill_steps} (pass {step // n_chunks + 1}) "
            f"kl {loss.item():.4f}  {rate:.2f} step/s  ETA {fmt_duration((args.distill_steps - step - 1) / rate)}",
            end="", flush=True,
        )
    print()
    del teacher_logits, state
    model.eval()
    torch.cuda.empty_cache()

    print(f"\n[{ts()}] === post-distillation probes (fresh state, no context) ===")
    n_match = 0
    pass_sum = 0.0
    rung_hits = 0
    deltas: list[float] = []
    for i, fact in enumerate(facts):
        rungs = cue_rungs(fact, user_open, asst_open)
        prompt = encode(rungs[0][0])
        target = encode(" " + fact.code)
        greedy = tokenizer.decode(generate(model, prompt, None, args.gen_tokens, 0.0)[0].cpu())
        matched = exact_match(greedy, fact.code, stops)

        batch = prompt.expand(args.pass_k, -1).contiguous()
        samples = [tokenizer.decode(row.cpu()) for row in generate(model, batch, None, args.gen_tokens, args.temperature)]
        pk = pass_at_k(samples, fact.code)

        def probe(rung: tuple[str, str], code: str = fact.code) -> bool:
            gen = tokenizer.decode(generate(model, encode(rung[0]), None, args.gen_tokens, 0.0)[0].cpu())
            return exact_match(rung[1] + gen, code, stops)

        # Rung 1 is the greedy probe already run above; reusing it keeps the
        # ladder consistent with `matched` instead of re-sampling it.
        later = 0 if matched else first_success(rungs[1:], probe)
        rung = 1 if matched else (later + 1 if later else 0)
        lp = target_logprob(model, prompt, target, None)
        delta = lp - float(pre[i]["logprob"])

        n_match += matched
        pass_sum += pk
        rung_hits += rung > 0
        deltas.append(delta)
        emit({
            "phase": "post", "fact": fact.entity, "category": fact.category, "code": fact.code,
            "greedy": greedy, "match": matched, "pass_at_k": pk, "k": args.pass_k,
            "rung": rung, "logprob_pre": pre[i]["logprob"], "logprob_post": lp, "logprob_delta": delta,
            "samples": samples,
        })
        print(
            f"[{ts()}]  post {i + 1}/{len(facts)} {fact.entity:<11} "
            f"greedy {'HIT ' if matched else 'miss'}  pass@{args.pass_k} {pk:.2f}  rung {rung}  "
            f"dlogp {delta:+.3f}  | running: match {n_match / (i + 1):.2f} "
            f"pass@k {pass_sum / (i + 1):.2f} any-rung {rung_hits / (i + 1):.2f}",
            flush=True,
        )
        if not matched and pk > 0:
            print(f"[{ts()}]      sampled hit: {next(s for s in samples if contains_code(s, fact.code))!r}")

    match_rate = n_match / len(facts)
    mean_delta = sum(deltas) / len(deltas)
    result = verdict(match_rate, mean_delta)
    summary = {
        "phase": "verdict", "verdict": result, "model": model_name, "checkpoint": args.checkpoint,
        "n_facts": len(facts), "filler_tokens": args.filler_tokens, "transcript_tokens": transcript.shape[1],
        "distill_steps": args.distill_steps, "lr": args.lr, "kl_temp": args.kl_temp, "seed": args.seed,
        "in_context_match_rate": ctx_rate, "fresh_floor_match_rate": floor_rate,
        "post_match_rate": match_rate, "post_pass_at_k": pass_sum / len(facts),
        "any_rung_rate": rung_hits / len(facts), "mean_logprob_delta": mean_delta,
        "pass_threshold": PASS_MATCH_RATE, "underpowered_threshold": UNDERPOWERED_DELTA_NATS,
        "memory_injection": has_memory and not args.no_memory,
        "fresh_state_replay": args.fresh_state_replay,
    }
    emit(summary)
    out_file.close()

    print(f"\n[{ts()}] === summary ===")
    print(f"[{ts()}]   in-context control : {ctx_rate:.3f}")
    print(f"[{ts()}]   fresh-state floor  : {floor_rate:.3f}")
    print(f"[{ts()}]   post greedy match  : {match_rate:.3f}  (pass threshold {PASS_MATCH_RATE})")
    print(f"[{ts()}]   post pass@{args.pass_k:<9}: {pass_sum / len(facts):.3f}")
    print(f"[{ts()}]   any cue rung       : {rung_hits / len(facts):.3f}")
    print(f"[{ts()}]   mean logprob delta : {mean_delta:+.4f} nats/token "
          f"(underpowered threshold {UNDERPOWERED_DELTA_NATS})")
    print(f"[{ts()}] VERDICT: {result}   -> {out_path}")


if __name__ == "__main__":
    main()
