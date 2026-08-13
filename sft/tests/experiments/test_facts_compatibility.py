import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2]))

import consolidation_null
from experiments import facts


def test_consolidation_null_reexports_fact_primitives():
    names = (
        "Fact",
        "build_facts",
        "build_turns",
        "cue_rungs",
        "exact_match",
        "render_turns",
    )

    assert all(getattr(consolidation_null, name) is getattr(facts, name) for name in names)
