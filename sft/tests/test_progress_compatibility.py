import consolidation_null
import progress


def test_consolidation_null_reexports_progress_helpers():
    assert consolidation_null.ts is progress.ts
    assert consolidation_null.fmt_duration is progress.fmt_duration
