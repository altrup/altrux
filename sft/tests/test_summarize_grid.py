"""The summarizer's registered-invariant check: every cell of a seed must have
distilled the same wake transcript and the same dream (DISCUSSION-20260806
sec 2 -- the 08-06 grid lost this in the harness->driver composition and
nothing noticed)."""

import json

import pytest

from summarize_grid import check_hashes, check_init_adapter, curve_rows, load_cells


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
