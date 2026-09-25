import json

import pytest

from experiments.lama_ckl.upstream import (
    RELEASE_FILES,
    adapt_single_gpu_script,
    prepare_single_gpu_smoke,
    reproduction_summary,
    verify_release,
)


def test_verify_release_checks_count_schema_and_hash(tmp_path):
    path = tmp_path / "variant.jsonl"
    row = {
        "uuid": "id-1",
        "relation_code": "P1",
        "subject": "Ada",
        "object": "London",
        "evidence": "Ada lived in London.",
        "task_descriptive": "Ada lives in London.",
        "invariant": False,
    }
    path.write_text(json.dumps(row) + "\n")
    files = {
        "variant.jsonl": {
            "rows": 1,
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        }
    }

    report = verify_release(tmp_path, files)

    assert report["variant.jsonl"]["rows"] == 1
    assert report["variant.jsonl"]["invariants"]["malformed"] == 0


def test_verify_release_rejects_changed_artifact(tmp_path):
    path = tmp_path / "variant.jsonl"
    path.write_text("{}\n")

    with pytest.raises(ValueError, match="sha256"):
        verify_release(tmp_path, {"variant.jsonl": {"rows": 1, "sha256": "0" * 64}})


def test_release_manifest_pins_the_four_official_files():
    assert RELEASE_FILES["variant.jsonl"]["rows"] == 500
    assert RELEASE_FILES["invariant_descriptive.jsonl"]["rows"] == 500
    assert RELEASE_FILES["invariant_schematic.jsonl"]["rows"] == 500
    assert RELEASE_FILES["train_attention_traindata.jsonl"]["rows"] == 4166


def test_reproduction_summary_uses_first_peak_and_checks_frozen_tolerance():
    curve = [
        {"epoch": 15, "to_learn_accuracy": 0.10, "not_to_forget_accuracy": 0.83},
        {"epoch": 16, "to_learn_accuracy": 0.115, "not_to_forget_accuracy": 0.8174},
        {"epoch": 17, "to_learn_accuracy": 0.115, "not_to_forget_accuracy": 0.80},
    ]

    result = reproduction_summary(curve)

    assert result == {
        "top_accuracy": 0.115,
        "epoch": 16,
        "not_to_forget_accuracy": 0.8174,
        "total_knowledge": 0.9324,
        "passes_gate": True,
    }


def test_single_gpu_adapter_preserves_global_batch_and_builds_one_step_smoke():
    source = (
        "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=12355 python evaluation_run.py \\\n"
        "--gpu_train_batch_size=8 \\\n"
        "--train_grad_accum_step=1 \\\n"
        "--max_epochs=30 \\\n"
        '--train_data="./data/LAMA_ckl/variant.jsonl" \\\n'
        '--eval_data_changed="./data/LAMA_ckl/variant.jsonl" \\\n'
        '--eval_data_unchanged="./data/LAMA_ckl/invariant_descriptive.jsonl" \\\n'
        '--add_to_title="" \\\n'
        '--bf16="true"\n'
    )

    full = adapt_single_gpu_script(source)
    smoke = adapt_single_gpu_script(source, smoke=True)

    assert "CUDA_VISIBLE_DEVICES=0 MASTER_PORT" in full
    assert "--gpu_train_batch_size=8" in full
    assert "--train_grad_accum_step=8" in full
    assert "--max_epochs=30" in full
    assert "--max_epochs=1" in smoke
    assert "gh200_smoke/variant.jsonl" in smoke
    assert "gh200_smoke/invariant_descriptive.jsonl" in smoke
    assert '--add_to_title="gh200_smoke"' in smoke


def test_single_gpu_adapter_rejects_unexpected_upstream_topology():
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        adapt_single_gpu_script("CUDA_VISIBLE_DEVICES=0 python evaluation_run.py")


def test_prepare_single_gpu_smoke_uses_exactly_64_official_rows(tmp_path):
    source = tmp_path / "data" / "LAMA_ckl"
    source.mkdir(parents=True)
    rows = [
        json.dumps(
            {
                "uuid": f"id-{index}",
                "relation_code": "P1",
                "subject": f"subject-{index}",
                "object": f"object-{index}",
                "evidence": f"subject-{index} maps to object-{index}.",
                "task_descriptive": f"subject-{index} maps to object-{index}.",
                "invariant": False,
            }
        )
        for index in range(65)
    ]
    for name in ("variant.jsonl", "invariant_descriptive.jsonl"):
        (source / name).write_text("\n".join(rows) + "\n")

    report = prepare_single_gpu_smoke(tmp_path)

    assert report == {"variant.jsonl": 64, "invariant_descriptive.jsonl": 64}
    assert len((source / "gh200_smoke" / "variant.jsonl").read_text().splitlines()) == 64
