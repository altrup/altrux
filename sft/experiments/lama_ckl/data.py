"""Official-notebook LAMA source conversion and model-conditioned sampling."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

INVARIANT_RELATIONS = frozenset(
    {
        "P19",
        "P20",
        "P279",
        "P37",
        "P449",
        "P47",
        "P138",
        "P364",
        "P527",
        "P176",
        "P27",
        "P407",
        "P30",
        "P178",
        "P1376",
        "P131",
        "P1412",
        "P17",
        "P276",
        "P937",
        "P140",
        "P103",
        "P190",
        "P1001",
        "P495",
        "P36",
        "P740",
        "P361",
    }
)


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    for line in path.read_text().splitlines():
        yield json.loads(line)


def build_candidates(lama_root: str | Path) -> Iterator[dict[str, object]]:
    """Yield the candidates produced by the pinned released notebook semantics."""
    root = Path(lama_root)
    relations = {str(row["relation"]): row for row in _read_jsonl(root / "relations.jsonl")}
    for path in sorted((root / "TREx").glob("P*.jsonl"), key=lambda item: int(item.stem[1:])):
        relation_code = path.stem
        relation = relations[relation_code]
        for source in _read_jsonl(path):
            evidences = source.get("evidences")
            if not isinstance(evidences, list) or not evidences:
                continue
            masked = max(
                (
                    str(item["masked_sentence"])
                    for item in evidences
                    if isinstance(item, dict) and isinstance(item.get("masked_sentence"), str)
                ),
                key=len,
                default="",
            )
            subject, obj = str(source["sub_label"]), str(source["obj_label"])
            evidence = masked.replace("[MASK]", obj)
            if len(evidence) <= 200 or subject not in evidence or obj not in evidence:
                continue
            template = str(relation["template"])
            yield {
                "relation_code": relation_code,
                "uuid": str(source["uuid"]),
                "task_descriptive": template.replace("[X]", subject).replace("[Y]", obj),
                "task_schematic": (
                    f"Guess the object. \n  subject is {subject} , relation is "
                    f"{relation['label']} , object is {obj}"
                ),
                "subject": subject,
                "relation_label": str(relation["label"]),
                "object": obj,
                "masked_evidence": masked,
                "evidence": evidence,
                "position": masked.index("[MASK]") / len(masked),
                "evidence_length": len(evidence),
                "invariant": relation_code in INVARIANT_RELATIONS,
            }


def select_split(
    rows: Sequence[dict[str, object]],
    size: int = 500,
    rng: random.Random | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Apply the released zero/one rules and sample the two frozen cohorts."""
    rng = rng or random.Random(42)
    p530 = 0
    learned: list[dict[str, object]] = []
    retained: list[dict[str, object]] = []
    for row in rows:
        scores = row.get("scores")
        if not isinstance(scores, dict):
            continue
        descriptive, schematic = scores.get("descriptive"), scores.get("schematic")
        if not isinstance(descriptive, (float, int)) or not isinstance(schematic, (float, int)):
            continue
        if row.get("invariant") is True and float(descriptive) == 1.0:
            retained.append(row)
        elif (
            row.get("invariant") is False and float(descriptive) == 0.0 and float(schematic) == 0.0
        ):
            if row.get("relation_code") == "P530":
                p530 += 1
                if p530 > 130:
                    continue
            learned.append(row)
    if len(learned) < size or len(retained) < size:
        raise ValueError(
            f"zero/one candidate floor failed: to_learn={len(learned)}, "
            f"not_to_forget={len(retained)}, need={size}"
        )
    return rng.sample(learned, size), rng.sample(retained, size)


def write_jsonl(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


__all__ = ["INVARIANT_RELATIONS", "build_candidates", "select_split", "write_jsonl"]
