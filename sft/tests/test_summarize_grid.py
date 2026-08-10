"""The summarizer's registered-invariant check: every cell of a seed must have
distilled the same wake transcript and the same dream (DISCUSSION-20260806
sec 2 -- the 08-06 grid lost this in the harness->driver composition and
nothing noticed)."""

import json

import pytest

from summarize_grid import (
    apply_floor,
    check_hashes,
    check_init_adapter,
    curve_rows,
    load_cells,
    main,
)


def _cell(path, arm, transcript_sha, dream_sha, wave2_dream_sha=None, init_adapter_sha256=None):
    rows = [
        {"phase": "cache", "wave": 1, "arm": arm, "seed": 1234,
         "transcript_sha": transcript_sha, "dream_sha": dream_sha},
        {"phase": "dream", "wave": 1, "arm": arm, "rehearsal_fraction": 0.2,
         "needle_counts": {}, "bound_cov": 3, "misbound": 1},
        {"phase": "sleep", "wave": 1, "arm": arm, "token_gradients": 800},
        {"phase": "probe", "wave": 1, "arm": arm, "step": 800, "fact": "osprey", "code": "1 2",
         "match": False, "logprob_delta": 0.5, "paraphrase_rate": 0.25, "margin": 1.5,
         "margin_install": True},
        {"phase": "in_context", "wave": 1, "arm": arm, "fact": "osprey", "match": True},
        {"phase": "locality", "wave": 1, "arm": arm, "ppl_delta": 0.1, "lost": 0, "items": 24},
        {"phase": "done", "arm": arm, "seed": 1234},
    ]
    if wave2_dream_sha:
        rows.append({"phase": "cache", "wave": 2, "arm": arm, "seed": 1234,
                     "transcript_sha": transcript_sha, "dream_sha": wave2_dream_sha})
    for row in rows:
        row["init_adapter_sha256"] = init_adapter_sha256
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_matching_hashes_pool(tmp_path):
    _cell(tmp_path / "g2_replay_s1234.jsonl", "replay", "aa", "bb")
    _cell(tmp_path / "g2_drain_s1234.jsonl", "drain", "aa", "bb")

    cells = load_cells(sorted(str(p) for p in tmp_path.glob("g2_*.jsonl")))

    assert len(cells) == 2
    check_hashes(cells)


def test_a_dream_hash_mismatch_refuses_to_pool(tmp_path):
    _cell(tmp_path / "g2_replay_s1234.jsonl", "replay", "aa", "bb")
    _cell(tmp_path / "g2_drain_s1234.jsonl", "drain", "aa", "cc")

    cells = load_cells(sorted(str(p) for p in tmp_path.glob("g2_*.jsonl")))

    with pytest.raises(SystemExit):
        check_hashes(cells)


def test_a_transcript_hash_mismatch_refuses_to_pool(tmp_path):
    _cell(tmp_path / "g2_replay_s1234.jsonl", "replay", "aa", "bb")
    _cell(tmp_path / "g2_drain_s1234.jsonl", "drain", "zz", "bb")

    with pytest.raises(SystemExit):
        check_hashes(load_cells(sorted(str(p) for p in tmp_path.glob("g2_*.jsonl"))))


def test_the_check_is_scoped_to_wave_one(tmp_path):
    """Wave-2 dreams legitimately differ per arm in the multi-sleep grid --
    recorded, never asserted."""
    _cell(tmp_path / "g2_replay_s1234.jsonl", "replay", "aa", "bb", wave2_dream_sha="w2a")
    _cell(tmp_path / "g2_drain_s1234.jsonl", "drain", "aa", "bb", wave2_dream_sha="w2b")

    check_hashes(load_cells(sorted(str(p) for p in tmp_path.glob("g2_*.jsonl"))))


def _cells(tmp_path):
    return load_cells(sorted(str(p) for p in tmp_path.glob("g2_*.jsonl")))


def test_arms_warm_started_from_the_same_checkpoint_pool(tmp_path):
    _cell(tmp_path / "g2_replay_s1234.jsonl", "replay", "aa", "bb", init_adapter_sha256="ff")
    _cell(tmp_path / "g2_drain_s1234.jsonl", "drain", "aa", "bb", init_adapter_sha256="ff")

    assert check_init_adapter(_cells(tmp_path)) == "ff"


def test_arms_from_different_warm_starts_refuse_to_pool(tmp_path):
    _cell(tmp_path / "g2_replay_s1234.jsonl", "replay", "aa", "bb", init_adapter_sha256="ff")
    _cell(tmp_path / "g2_drain_s1234.jsonl", "drain", "aa", "bb", init_adapter_sha256="ee")

    with pytest.raises(SystemExit, match="warm start"):
        check_init_adapter(_cells(tmp_path))


def test_a_warm_started_arm_never_pools_with_a_cold_one(tmp_path):
    _cell(tmp_path / "g2_replay_s1234.jsonl", "replay", "aa", "bb", init_adapter_sha256="ff")
    _cell(tmp_path / "g2_drain_s1234.jsonl", "drain", "aa", "bb")

    with pytest.raises(SystemExit):
        check_init_adapter(_cells(tmp_path))


def test_a_grid_with_no_warm_start_pools_as_before(tmp_path):
    _cell(tmp_path / "g2_replay_s1234.jsonl", "replay", "aa", "bb")
    _cell(tmp_path / "g2_drain_s1234.jsonl", "drain", "aa", "bb")

    assert check_init_adapter(_cells(tmp_path)) is None


def test_cells_carry_the_new_columns(tmp_path):
    path = _cell(tmp_path / "g2_replay_s1234.jsonl", "replay", "aa", "bb")
    cell = load_cells([str(path)])[0]

    assert cell["bound_cov"] == 3 and cell["misbound"] == 1
    assert cell["token_gradients"] == 800
    assert cell["margin_install"] == 1
    assert cell["seed"] == "1234"


def _rows(path):
    return [json.loads(line) for line in open(path)]


def _rewrite(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_the_cell_carries_its_erase_operator(tmp_path):
    path = _cell(tmp_path / "g2_drain_s1234.jsonl", "drain", "aa", "bb")
    rows = _rows(path)
    for row in rows:
        row["erase_op"] = "raw"
    _rewrite(path, rows)

    assert load_cells([str(path)])[0]["erase_op"] == "raw"


def test_a_cell_that_changed_erase_operator_mid_run_is_refused(tmp_path):
    path = _cell(tmp_path / "g2_drain_s1234.jsonl", "drain", "aa", "bb")
    rows = _rows(path)
    for i, row in enumerate(rows):
        row["erase_op"] = "raw" if i else "deflated"
    _rewrite(path, rows)

    with pytest.raises(SystemExit, match="erase"):
        load_cells([str(path)])


def test_cells_with_different_erase_operators_still_pool(tmp_path):
    """The picker grid deliberately runs raw and deflated cells side by side."""
    for arm, op in (("drain_raw", "raw"), ("drain_deflated", "deflated")):
        path = _cell(tmp_path / f"g2_{arm}_s1234.jsonl", arm, "aa", "bb")
        rows = _rows(path)
        for row in rows:
            row["erase_op"] = op
        _rewrite(path, rows)

    assert {c["erase_op"] for c in _cells(tmp_path)} == {"raw", "deflated"}


def test_a_failed_b3_equivalence_is_fatal_to_the_summary(tmp_path):
    path = _cell(tmp_path / "g2_b3fused_s1234.jsonl", "b3-fused", "aa", "bb")
    rows = _rows(path)
    rows.append({"phase": "equivalence", "wave": 2, "arm": "b3-fused", "pass": 1,
                 "abs_diff": 0.5, "equivalent": False})
    _rewrite(path, rows)

    with pytest.raises(SystemExit, match="equivalen"):
        load_cells([str(path)])


def test_a_passing_b3_equivalence_scores_normally(tmp_path):
    path = _cell(tmp_path / "g2_b3fused_s1234.jsonl", "b3-fused", "aa", "bb")
    rows = _rows(path)
    rows.append({"phase": "equivalence", "wave": 1, "arm": "b3-fused", "pass": 1,
                 "abs_diff": 0.0, "equivalent": True})
    _rewrite(path, rows)

    assert len(load_cells([str(path)])) == 1


def _curve_cell(path, arm, points, floor_only_final=False):
    """points: {step: (margin, ppl_delta)} for the single fact 'osprey'."""
    rows = [{"phase": "cache", "wave": 1, "arm": arm, "seed": 1234,
             "transcript_sha": "aa", "dream_sha": "bb"},
            {"phase": "done", "arm": arm, "seed": 1234}]
    for step, (margin, dppl) in points.items():
        rows.append({"phase": "probe", "wave": 1, "arm": arm, "step": step, "fact": "osprey",
                     "code": "1 2", "match": False, "logprob_delta": 0.0, "paraphrase_rate": 0.0,
                     "margin": margin, "margin_install": False})
        rows.append({"phase": "locality", "wave": 1, "arm": arm, "step": step,
                     "ppl_delta": dppl, "lost": 0, "items": 24})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_curves_correct_each_probe_step_against_the_floor_at_the_same_step(tmp_path):
    _curve_cell(tmp_path / "g2_nosleep_s1234.jsonl", "nosleep", {200: (1.0, 0.0), 400: (2.0, 0.0)})
    _curve_cell(tmp_path / "g2_drain_s1234.jsonl", "drain", {200: (3.0, 0.1), 400: (5.0, 0.3)})

    rows = curve_rows(_cells(tmp_path))
    drain = [r for r in rows if r["arm"] == "drain"]

    assert [(r["step"], r["dmargin"], r["dppl"]) for r in drain] == [(200, 2.0, 0.1), (400, 3.0, 0.3)]


def test_curves_fall_back_to_the_final_floor_when_the_step_is_missing(tmp_path):
    _curve_cell(tmp_path / "g2_nosleep_s1234.jsonl", "nosleep", {400: (2.0, 0.0)})
    _curve_cell(tmp_path / "g2_drain_s1234.jsonl", "drain", {200: (3.0, 0.1), 400: (5.0, 0.3)})

    drain = [r for r in curve_rows(_cells(tmp_path)) if r["arm"] == "drain"]

    assert [(r["step"], r["dmargin"]) for r in drain] == [(200, 1.0), (400, 3.0)]


# ---- ladder cells: the seed token is the seed, the rung is part of the arm --


def test_a_ladder_cell_parses_its_seed_and_keeps_its_rung_in_the_arm(tmp_path):
    """`lad_A_s1234_d3200` parsed its seed as `1234_d3200`, which matched no
    floor -- the root cause of the run notes' `--fallback final_floor` no-rows
    report. The rung has to stay in the arm, or d800 and d3200 pool as one."""
    path = _curve_cell(tmp_path / "lad_A_s1234_d3200.jsonl", "lad_A", {3200: (5.0, 0.3)})

    cell = load_cells([str(path)])[0]

    assert cell["seed"] == "1234" and cell["arm"] == "lad_A_d3200"


def test_a_ladder_cell_is_floor_corrected_against_its_seeds_no_sleep_cell(tmp_path):
    """The floor sits at step 800 and the rung's probe at 3200, so this is
    exactly the case `fallback=final_floor` exists for -- and it emitted no
    rows because the seeds never matched."""
    _curve_cell(tmp_path / "nosleep_s1234.jsonl", "nosleep", {800: (2.0, 0.0)})
    _curve_cell(tmp_path / "lad_A_s1234_d3200.jsonl", "lad_A", {3200: (5.0, 0.3)})

    cells = load_cells(sorted(str(p) for p in tmp_path.glob("*.jsonl")))
    ladder = [r for r in curve_rows(cells) if r["arm"] == "lad_A_d3200"]

    assert [(r["step"], r["dmargin"]) for r in ladder] == [(3200, 3.0)]
    apply_floor(cells)
    assert next(c["dmargin"] for c in cells if c["arm"] == "lad_A_d3200") == 3.0


# ---- multi-dream cells (sec 2.10.4) ---------------------------------------


def _set_cell(path, arm, set_sha="s1", dreams=2, transcript_sha="aa"):
    """A dream-set cell: no `dream_sha`, no rehearsal fraction, one `dream`
    record per boundary carrying its epoch and dream index."""
    rows = [{"phase": "cache", "wave": 1, "arm": arm, "seed": 1234, "transcript_sha": transcript_sha,
             "dreams": dreams, "set_sha": set_sha},
            {"phase": "sleep", "wave": 1, "arm": arm, "dreams": dreams, "epochs": 1,
             "token_gradients": 1200},
            {"phase": "in_context", "wave": 1, "arm": arm, "fact": "osprey", "match": True},
            {"phase": "locality", "wave": 1, "arm": arm, "ppl_delta": 0.1, "lost": 0, "items": 24},
            {"phase": "done", "arm": arm, "seed": 1234}]
    for i in range(dreams):
        rows.append({"phase": "dream", "wave": 1, "arm": arm, "dream": i, "epoch": 0, "step": i + 1,
                     "dream_sha": f"d{i}", "tokens": 300, "stop_reason": "eoc",
                     "gated_positions": 40, "basis_rank": [2, 2],
                     "bound_by_fact": {"osprey": 1 if i == 0 else 0, "heron": 1},
                     "misbound_by_fact": {"osprey": 0, "heron": 1}, "bound_cov": 2 - i})
        rows.append({"phase": "probe", "wave": 1, "arm": arm, "step": i + 1, "fact": "osprey",
                     "code": "1 2", "match": False, "logprob_delta": 0.5, "paraphrase_rate": 0.25,
                     "margin": 1.5, "margin_install": True})
    for row in rows:
        row["dream_set_sha"] = set_sha
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_a_dream_set_cell_scores_with_aggregate_coverage(tmp_path):
    path = _set_cell(tmp_path / "g3_b4-raw_s1234.jsonl", "b4-raw")

    cell = load_cells([str(path)])[0]

    assert cell["dreams"] == 2 and cell["set_sha"] == "s1"
    assert cell["bound_cov"] == 2  # osprey in dream 0, heron in both -- aggregate, not per dream
    assert cell["misbound"] == 2
    assert cell["rehearse"] is None  # no single-dream rehearsal fraction exists for a set
    assert cell["token_gradients"] == 1200


def test_cells_that_distilled_different_dream_sets_refuse_to_pool(tmp_path):
    _set_cell(tmp_path / "g3_replay_s1234.jsonl", "replay", set_sha="s1")
    _set_cell(tmp_path / "g3_b4-raw_s1234.jsonl", "b4-raw", set_sha="s2")

    with pytest.raises(SystemExit, match="set_sha"):
        check_hashes(load_cells(sorted(str(p) for p in tmp_path.glob("g3_*.jsonl"))))


def test_dream_set_cells_that_share_a_set_pool(tmp_path):
    _set_cell(tmp_path / "g3_replay_s1234.jsonl", "replay")
    _set_cell(tmp_path / "g3_b4-raw_s1234.jsonl", "b4-raw")

    cells = load_cells(sorted(str(p) for p in tmp_path.glob("g3_*.jsonl")))

    check_hashes(cells)
    assert len(cells) == 2


def test_the_whole_summary_runs_on_a_dream_set_grid(tmp_path, capsys):
    _set_cell(tmp_path / "g3_replay_s1234.jsonl", "replay")
    _set_cell(tmp_path / "g3_b4-raw_s1234.jsonl", "b4-raw")

    main(str(tmp_path / "g3_*.jsonl"))

    out = capsys.readouterr().out
    assert "b4-raw" in out and "replay" in out
