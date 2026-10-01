"""Experiment: state-erasure

Dream cache construction and sleep/wave orchestration."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

from experiments.dreams.cache import (
    binding_coverage,
    dream_bases,
    load_dream_cache,
    pilot_path,
    save_dream_cache,
    sidecar_path,
    write_dream_set_sidecar,
    write_dream_sidecar,
)
from experiments.dreams.distillation import (
    distill_counterfactual,
    distill_dream_set,
    distill_fused,
    distill_live,
    distill_replay,
    distill_sft,
    dream_from_cached,
    erase_state,
    erased_start,
    erased_start_scaled,
    scored_keep,
    sft_steps,
    spine_states,
)
from experiments.dreams.generation import (
    copy_state,
    dream_generation_seed,
    dream_seed_text,
    rehearsal_fraction,
    state_to,
    teacher_dream,
)
from experiments.dreams.probes import (
    battery_read_queries,
    blank_state_logits,
    dream_is_degenerate,
    report_dream,
    report_dream_set,
)
from experiments.dreams.types import CachedDream, Dream, DreamCache, DreamSetCache, token_sha
from experiments.erasure.gating import VARIANTS, gated_positions, state_divergence
from experiments.erasure.pilot import PilotCapture, PilotDream, scheme_weights
from experiments.erasure.probe import group_by_layer
from experiments.facts import Fact, build_distractors
from experiments.inference import run_chunks
from progress import fmt_duration, ts

USER_CUE = "{user} What is the code for the {entity}?"
DREAM_RETRIES = 2
B4_ARMS = {f"b4-{variant}": variant for variant in VARIANTS}
B4_ARMS["b4-sigma"] = "raw"
FUSED_ARMS = ("b2-fused-detached", "b2-fused-deep", "b3-fused")


def build_cues(seen, encode, user_open: str, asst_open: str) -> list[list[int]]:
    return [
        encode(
            f"{USER_CUE.format(user=user_open, entity=f.entity)}"
            f"{asst_open} The code for the {f.entity} is"
        )[0].tolist()
        for _, f in seen
    ]


def validate_wave_args(args) -> None:
    if args.waves > 1 and not getattr(args, "wave_teacher", None):
        raise SystemExit(
            "--waves > 1 needs --wave-teacher base|current: the registered protocol is `current` "
            "(sec 3.2), `base` is the one-seed drift-contribution control. Register the choice "
            "before running multi-sleep."
        )


def validate_live_wake_args(args) -> None:
    if not getattr(args, "live_wake", False):
        return
    for name, message in (
        (
            "wake_plan",
            "--live-wake requires --wake-plan; wake length and injection positions are not defaults",
        ),
        ("user_generator_command", "--live-wake requires --user-generator-command"),
        ("wake_scenarios", "--live-wake requires --wake-scenarios"),
    ):
        if not getattr(args, name, None):
            raise SystemExit(message)
    missing = [
        name
        for name in ("user_generator_provider", "user_generator_model", "user_generator_version")
        if not getattr(args, name, None)
    ]
    if missing:
        raise SystemExit("--live-wake requires pinned user-generator provider, model, and version")


def load_live_wake_scenarios(path: str | Path, waves: int) -> list[str]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read --wake-scenarios {path}: {exc}") from exc
    if (
        not isinstance(value, list)
        or len(value) != waves
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise SystemExit(f"--wake-scenarios requires exactly {waves} non-empty scenario strings")
    return value


def render_live_wake_transcript(artifact: dict[str, object], user_open: str, asst_open: str) -> str:
    turns = artifact.get("turns")
    if not isinstance(turns, list):
        raise SystemExit("live wake artifact has no turns")
    rendered: list[str] = []
    for turn in turns:
        if (
            not isinstance(turn, dict)
            or not isinstance(turn.get("user"), str)
            or not isinstance(turn.get("assistant"), str)
        ):
            raise SystemExit("live wake artifact has malformed turns")
        rendered.append(f"{user_open} {turn['user']}{asst_open} {turn['assistant']}")
    return "".join(rendered)


def generate_wave_dream(
    model,
    args,
    carried,
    facts,
    encode,
    decode,
    tokenizer,
    user_open,
    asst_open: str,
    teacher: str,
    stop_id: int | None,
) -> Dream:
    seed_ids = encode(dream_seed_text(asst_open, args.dream_prompt))
    needles = [f.entity for f in facts] + [f.code for f in facts]
    cues = (
        build_cues([(0, f) for f in facts], encode, user_open, asst_open) if args.cue_every else []
    )
    model.eval()
    return teacher_dream(
        model,
        carried,
        seed_ids,
        args.dream_tokens,
        args.dream_temp,
        drain=False,
        decode_token=lambda i: decode([i]),
        needles=needles,
        cues=cues,
        cue_every=args.cue_every,
        cue_greedy=args.cue_greedy,
        frozen=(teacher == "base"),
        stop_id=stop_id,
        turn_id=tokenizer.eos_token_id,
    )


def r_matrix_row(scored: dict[str, dict[str, object]]) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for record in scored.values():
        if record.get("margin") is None:
            continue
        row = rows.setdefault(
            int(record["fact_wave"]), {"mean_margin": 0.0, "installs": 0, "facts": 0}
        )
        row["mean_margin"] += float(record["margin"])
        row["installs"] += int(bool(record["install"]))
        row["facts"] += 1
    for row in rows.values():
        row["mean_margin"] /= row["facts"]
    return rows


def r_matrix_rows(
    r: dict[tuple[int, int], dict[str, float]], waves: int
) -> list[list[float | None]]:
    return [
        [r[(i, j)]["mean_margin"] if (i, j) in r else None for j in range(1, waves + 1)]
        for i in range(1, waves + 1)
    ]


def cl_summary(r: dict[tuple[int, int], dict[str, float]], waves: int) -> dict[str, object]:
    deltas = [
        r[(i, waves)]["mean_margin"] - r[(i, i)]["mean_margin"]
        for i in range(1, waves)
        if (i, waves) in r and (i, i) in r
    ]
    peak = sum(
        max(v["installs"] for (wave, _), v in r.items() if wave == i)
        for i in sorted({i for i, _ in r})
    )
    return {
        "bwt": sum(deltas) / len(deltas) if deltas else None,
        "installs_final": sum(v["installs"] for (_, j), v in r.items() if j == waves),
        "installs_peak": peak,
        "waves": waves,
    }


def commit_erase(
    model,
    carried,
    seen,
    committed: set[str],
    encode,
    user_open: str,
    asst_open: str,
    verify,
    erase_op: str,
    emit,
    wave: int,
) -> list[str]:
    import torch

    fired: list[str] = []
    for _, fact in seen:
        if fact.entity in committed:
            continue
        margin, installed, *_ = verify(fact)
        record = {
            "phase": "commit",
            "wave": wave,
            "fact": fact.entity,
            "margin": margin,
            "committed": bool(installed),
        }
        if installed:
            prompt = encode(
                f"{USER_CUE.format(user=user_open, entity=fact.entity)}"
                f"{asst_open} The code for the {fact.entity} is"
            )
            model.c_capture = []
            with torch.no_grad():
                model(prompt, state=copy_state(carried))
            queries = group_by_layer(model.c_capture, len(model.layers))[-1]
            model.c_capture = None
            record["skipped_cone"] = erase_state(carried, queries, op=erase_op)
            committed.add(fact.entity)
            fired.append(fact.entity)
        emit(record)
        print(
            f"[{ts()}]  commit {fact.entity:<11} margin {margin:+7.3f} "
            f"{'ERASED' if installed else 'held (not installed)'}",
            flush=True,
        )
    return fired


def build_cache(
    model,
    args,
    cache_path: Path,
    transcript,
    facts,
    chunk_len,
    encode,
    decode,
    tokenizer,
    user_open,
    asst_open,
    stop_id: int | None,
    adapter_sha: str | None = None,
    wake_state=None,
) -> DreamCache:
    import torch

    wake_state = (
        run_chunks(model, transcript, None, chunk_len, "wake", keep_logits=False)[1]
        if wake_state is None
        else copy_state(wake_state)
    )
    seed_ids = encode(dream_seed_text(asst_open, args.dream_prompt))
    needles = [f.entity for f in facts] + [f.code for f in facts]
    cues = (
        build_cues([(1, f) for f in facts], encode, user_open, asst_open) if args.cue_every else []
    )
    model.eval()
    dream = teacher_dream(
        model,
        wake_state,
        seed_ids,
        args.dream_tokens,
        args.dream_temp,
        drain=False,
        decode_token=lambda i: decode([i]),
        needles=needles,
        cues=cues,
        cue_every=args.cue_every,
        cue_greedy=args.cue_greedy,
        frozen=adapter_sha is None,
        stop_id=stop_id,
        turn_id=tokenizer.eos_token_id,
    )
    cache = DreamCache(
        seed=args.seed,
        transcript_ids=[int(i) for i in transcript[0].tolist()],
        dream_ids=[int(i) for i in dream.tokens[0].tolist()],
        wake_state=state_to(wake_state, torch.device("cpu")),
        teacher_logits=dream.logits.cpu(),
        queries=[[c.cpu() for c in per_layer] for per_layer in dream.queries],
        token_texts=dream.token_texts,
        cue_flags=dream.cue_flags,
        distractors=build_distractors(facts, args.seed),
        facts=[(f.entity, f.category, f.code) for f in facts],
        stop_reason=dream.stop_reason,
        dream_prompt=args.dream_prompt,
        prefix_len=seed_ids.shape[1],
        generator=adapter_sha or "base",
    )
    save_dream_cache(cache, cache_path)
    sidecar = sidecar_path(cache_path)
    write_dream_sidecar(cache, sidecar)
    print(f"[{ts()}] wrote {cache_path} and {sidecar}")
    print(
        f"[{ts()}] transcript_sha {cache.transcript_sha}\n[{ts()}] dream_sha      {cache.dream_sha}"
    )
    report_dream(cache)
    return cache


def build_dream_set(
    model,
    args,
    cache_path: Path,
    transcript,
    facts,
    chunk_len,
    encode,
    decode,
    tokenizer,
    user_open,
    asst_open,
    stop_id: int | None,
    adapter_sha: str | None = None,
    battery=None,
    wake_state=None,
) -> DreamSetCache:
    import torch

    wake_state = (
        run_chunks(model, transcript, None, chunk_len, "wake", keep_logits=False)[1]
        if wake_state is None
        else copy_state(wake_state)
    )
    seed_ids = encode(dream_seed_text(asst_open, args.dream_prompt))
    prefix_len = seed_ids.shape[1]
    needles = [f.entity for f in facts] + [f.code for f in facts]
    cues = (
        build_cues([(1, f) for f in facts], encode, user_open, asst_open) if args.cue_every else []
    )
    model.eval()
    dreams: list[CachedDream] = []
    pilot = []
    started = time.time()
    regenerated = 0
    retries = DREAM_RETRIES
    for i in range(args.dreams):
        for attempt in range(retries + 1):
            gen_seed = dream_generation_seed(
                args.seed, i, attempt, getattr(args, "dream_seed_offset", 0)
            )
            torch.manual_seed(gen_seed)
            print(f"\n[{ts()}] --- dream {i + 1}/{args.dreams} (generation seed {gen_seed}) ---")
            # teacher_dream(
            dream = teacher_dream(
                model,
                wake_state,
                seed_ids,
                args.dream_tokens,
                args.dream_temp,
                drain=False,
                decode_token=lambda j: decode([j]),
                needles=needles,
                cues=cues,
                cue_every=args.cue_every,
                cue_greedy=args.cue_greedy,
                frozen=adapter_sha is None,
                stop_id=stop_id,
                turn_id=tokenizer.eos_token_id,
            )
            if not dream_is_degenerate("".join(dream.token_texts)):
                break
            regenerated += 1
        else:
            raise SystemExit(f"dream slot {i} stayed degenerate through {retries + 1} attempts")
        blank = blank_state_logits(model, dream.tokens, chunk_len, frozen=adapter_sha is None)
        divergence = state_divergence(dream.logits, blank)
        gate = gated_positions(divergence, args.gate_threshold, prefix_len, dream.cue_flags)
        family = getattr(args, "gate_family", "hard")
        weights = (
            None
            if family == "hard"
            else scheme_weights([float(divergence[t]) for t in gate], family)
        )
        spectra, ranks, bases = dream_bases(
            dream.queries, gate, wake_state, args.rank_rule, weights
        )
        dreams.append(
            CachedDream(
                dream_ids=[int(t) for t in dream.tokens[0].tolist()],
                token_texts=dream.token_texts,
                teacher_logits=dream.logits.cpu(),
                cue_flags=dream.cue_flags,
                prefix_len=prefix_len,
                stop_reason=dream.stop_reason,
                divergence=[float(d) for d in divergence],
                gate_positions=gate,
                queries=[[c.cpu() for c in dream.queries[t]] for t in gate],
                spectra=spectra,
                ranks=ranks,
                bases=bases,
            )
        )
        if args.pilot_capture:
            pilot.append(
                PilotDream(
                    dream_sha=dreams[-1].dream_sha,
                    token_texts=dream.token_texts,
                    divergence=[float(d) for d in divergence],
                    queries=[[c.half().cpu() for c in per_layer] for per_layer in dream.queries],
                    cue_flags=dream.cue_flags,
                    prefix_len=prefix_len,
                    stop_reason=dream.stop_reason,
                )
            )
        elapsed = time.time() - started
        print(
            f"[{ts()}]  dream {i + 1} cached; {fmt_duration(elapsed)} elapsed, ETA "
            f"{fmt_duration(elapsed / (i + 1) * (args.dreams - i - 1))}",
            flush=True,
        )
    cache = DreamSetCache(
        seed=args.seed,
        transcript_ids=[int(t) for t in transcript[0].tolist()],
        wake_state=state_to(wake_state, torch.device("cpu")),
        dreams=dreams,
        distractors=build_distractors(facts, args.seed),
        facts=[(f.entity, f.category, f.code) for f in facts],
        dream_seed_offset=getattr(args, "dream_seed_offset", 0),
        gate_family=getattr(args, "gate_family", "hard"),
        dream_prompt=args.dream_prompt,
        gate_threshold=args.gate_threshold,
        rank_rule=args.rank_rule,
        generator=adapter_sha or "base",
    )
    report_dream_set(cache, args.bind_min_dreams, args.rank_rule)
    save_dream_cache(cache, cache_path)
    sidecar = sidecar_path(cache_path)
    write_dream_set_sidecar(cache, sidecar)
    print(f"\n[{ts()}] wrote {cache_path} and {sidecar}")
    if args.pilot_capture:
        capture = PilotCapture(
            seed=args.seed,
            facts=cache.facts,
            wake_state=cache.wake_state,
            dreams=pilot,
            battery_queries=battery_read_queries(
                model, battery, wake_state, encode, len(model.layers)
            )
            if battery
            else {},
            gate_threshold=args.gate_threshold,
            rank_rule=args.rank_rule,
            set_sha=cache.set_sha,
        )
        torch.save(capture, pilot_path(cache_path))
    return cache


def run_dream_set_sleep(
    mode,
    model,
    opt,
    args,
    wave,
    wake_state,
    seen,
    cache: DreamSetCache,
    emit,
    periodic_probe,
    on_step,
    started: float,
) -> object:
    variant = B4_ARMS.get(mode)
    facts = [f for _, f in seen]
    probe_seconds = 0.0
    train_started = time.time()

    def on_boundary(index: int, epoch: int, step: int) -> None:
        nonlocal probe_seconds
        dream = cache.dreams[index]
        bound, misbound = binding_coverage("".join(dream.token_texts), facts)
        emit(
            {
                "phase": "dream",
                "wave": wave,
                "arm": mode,
                "dream": index,
                "epoch": epoch,
                "step": step,
                "dream_sha": dream.dream_sha,
                "tokens": len(dream.dream_ids),
                "stop_reason": dream.stop_reason,
                "gated_positions": len(dream.gate_positions),
                "basis_rank": [len(b) for b in dream.bases[variant]] if variant else None,
                "bound_by_fact": bound,
                "misbound_by_fact": misbound,
                "bound_cov": sum(v > 0 for v in bound.values()),
            }
        )
        if index % args.probe_every_dream:
            return
        print()
        at = time.time()
        periodic_probe(step)
        probe_seconds += time.time() - at

    tokens = distill_dream_set(
        model,
        opt,
        cache.dreams,
        wake_state,
        variant,
        args.dream_epochs,
        args.kl_temp,
        on_step,
        on_boundary,
        sigma_scaled=(mode == "b4-sigma"),
    )
    train_seconds = time.time() - train_started - probe_seconds
    print(
        f"\n[{ts()}] boundary probes {fmt_duration(probe_seconds)} against {fmt_duration(train_seconds)} of training"
    )
    emit(
        {
            "phase": "sleep",
            "wave": wave,
            "arm": mode,
            "dreams": len(cache.dreams),
            "epochs": args.dream_epochs,
            "steps": len(cache.dreams) * args.dream_epochs,
            "token_gradients": tokens,
            "probe_seconds": probe_seconds,
            "train_seconds": train_seconds,
            "seconds": time.time() - started,
        }
    )
    return wake_state


def run_sleep(
    mode,
    model,
    opt,
    args,
    wave,
    wake_state,
    transcript,
    seen,
    chunk_len,
    cache,
    encode,
    decode,
    tokenizer,
    user_open,
    asst_open,
    emit,
    periodic_probe,
    stop_id,
    verify=None,
    committed=None,
):
    import torch

    started = time.time()
    wave_facts = [f for w, f in seen if w == wave]
    dream_set = isinstance(cache, DreamSetCache) and mode not in ("sft-ref", "no-sleep")
    sft_step_count = sft_steps(transcript.shape[1], chunk_len)
    total = (
        len(cache.dreams) * args.dream_epochs
        if dream_set
        else args.dream_tokens
        if mode == "drain-live"
        else sft_step_count
        if mode == "sft-ref"
        else args.distill_steps
    )

    def on_step(step: int, loss: float) -> None:
        emit(
            {
                "phase": "kl",
                "wave": wave,
                "arm": mode,
                "step": step,
                "loss": loss,
                "accum_window": args.accum_window,
            }
        )
        rate = (step + 1) / (time.time() - started)
        print(
            f"\r[{ts()}]  {mode} step {step + 1}/{total} loss {loss:.4f} {rate:.2f} step/s "
            f"ETA {fmt_duration((total - step - 1) / rate)}",
            end="",
            flush=True,
        )
        if (
            args.probe_every
            and not dream_set
            and (step + 1) % args.probe_every == 0
            and step + 1 < total
        ):
            print()
            periodic_probe(step + 1)

    if dream_set:
        model.train()
        carried = run_dream_set_sleep(
            mode,
            model,
            opt,
            args,
            wave,
            wake_state,
            seen,
            cache,
            emit,
            periodic_probe,
            on_step,
            started,
        )
        model.eval()
        torch.cuda.empty_cache()
        return carried

    needles = [f.entity for _, f in seen] + [f.code for _, f in seen]
    dream = None
    if mode not in ("sft-ref", "drain-live"):
        if wave == 1:
            dream = dream_from_cached(cache, wake_state.ssm_states[0].device)
        else:
            dream = generate_wave_dream(
                model,
                args,
                wake_state,
                wave_facts,
                encode,
                decode,
                tokenizer,
                user_open,
                asst_open,
                teacher=args.wave_teacher,
                stop_id=stop_id,
            )
            emit(
                {
                    "phase": "cache",
                    "wave": wave,
                    "arm": mode,
                    "seed": args.seed,
                    "dream_sha": token_sha(dream.tokens[0].tolist()),
                    "dream_tokens": dream.tokens.shape[1],
                    "free_tokens": sum(not f for f in dream.cue_flags),
                    "cues_cover": [f.entity for f in wave_facts],
                    "dream_generator": "base" if args.wave_teacher == "base" else "student",
                }
            )

    model.train()
    if mode == "sft-ref":
        token_gradients = distill_sft(model, opt, transcript, sft_step_count, chunk_len, on_step)
        print()
        emit(
            {
                "phase": "sleep",
                "wave": wave,
                "arm": mode,
                "steps": sft_step_count,
                "token_gradients": token_gradients,
                "seconds": time.time() - started,
            }
        )
        model.eval()
        return None
    if mode == "drain-live":
        seed_ids = encode(dream_seed_text(asst_open, args.dream_prompt))
        carried, dream = distill_live(
            model,
            opt,
            wake_state,
            seed_ids,
            args.dream_tokens,
            args.dream_temp,
            args.kl_temp,
            args.accum_window,
            lambda i: decode([i]),
            needles,
            on_step,
            erase_op=args.erase_op,
        )
        token_gradients = args.dream_tokens
        print()
    else:
        keep = scored_keep(dream.cue_flags, dream.prefix_len) if dream.cue_flags else None
        fused_arms = FUSED_ARMS
        if mode in ("replay", "ce-on-dream"):
            replay_chunk = args.chunk_len or dream.tokens.shape[1]
            token_gradients = distill_replay(
                model,
                opt,
                dream,
                args.distill_steps,
                replay_chunk,
                args.kl_temp,
                on_step,
                fresh_state=args.fresh_state_replay,
                keep=keep,
                ce=mode == "ce-on-dream",
            )
            carried = None
        elif mode in fused_arms:
            generator_spine = None
            if mode == "b3-fused":
                with torch.no_grad():
                    generator_spine = spine_states(
                        model, dream.tokens, wake_state, args.spine_block
                    )
            token_gradients = distill_fused(
                model,
                opt,
                dream,
                wake_state,
                args.distill_steps,
                args.kl_temp,
                on_step,
                keep=keep,
                erase_op=args.erase_op,
                deep=mode == "b2-fused-deep",
                block=args.spine_block,
                cf_batch=args.cf_batch,
                frozen_spine=generator_spine,
                check=lambda record: emit(
                    {"phase": "equivalence", "wave": wave, "arm": mode, **record}
                ),
            )
            carried = wake_state
        else:
            token_gradients, drained = distill_counterfactual(
                model,
                opt,
                dream,
                wake_state,
                args.distill_steps,
                args.kl_temp,
                args.accum_window,
                in_place=mode == "drain",
                deep=args.deep,
                on_step=on_step,
                keep=keep,
                erase_op=args.erase_op,
            )
            carried = drained if mode == "drain" else wake_state
        print()

    fraction, counts = rehearsal_fraction(dream.token_texts, needles)
    facts = [f for _, f in seen]
    bound, misbound = binding_coverage("".join(dream.token_texts), facts)
    emit(
        {
            "phase": "dream",
            "wave": wave,
            "arm": mode,
            "tokens": len(dream.token_texts),
            "rehearsal_fraction": fraction,
            "needle_counts": counts,
            "skipped_cone": dream.skipped_cone,
            "bound_cov": sum(v > 0 for v in bound.values()),
            "misbound": sum(misbound.values()),
            "bound_by_fact": bound,
            "misbound_by_fact": misbound,
            "temperature": args.dream_temp,
            "prompt": args.dream_prompt,
            "decoded": "".join(dream.token_texts),
        }
    )
    emit(
        {
            "phase": "sleep",
            "wave": wave,
            "arm": mode,
            "steps": args.distill_steps,
            "token_gradients": token_gradients,
            "deep": args.deep,
            "seconds": time.time() - started,
        }
    )
    if mode == "counterfactual-commit":
        if verify is None:
            raise SystemExit(
                "arm counterfactual-commit needs a fresh-state margin check to commit against"
            )
        model.eval()
        commit_erase(
            model,
            carried,
            seen,
            set() if committed is None else committed,
            encode,
            user_open,
            asst_open,
            verify,
            args.erase_op,
            emit,
            wave,
        )
    model.eval()
    torch.cuda.empty_cache()
    return carried


__all__ = [
    "build_cache",
    "build_cues",
    "build_dream_set",
    "cl_summary",
    "commit_erase",
    "generate_wave_dream",
    "load_live_wake_scenarios",
    "r_matrix_row",
    "r_matrix_rows",
    "render_live_wake_transcript",
    "run_dream_set_sleep",
    "run_sleep",
    "validate_live_wake_args",
    "validate_wave_args",
]
