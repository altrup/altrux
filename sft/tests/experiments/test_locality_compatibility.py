import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2]))

import probes_common
from experiments import locality


PUBLIC_NAMES = (
    "BATTERY_CANDIDATES",
    "BatteryItem",
    "CalibrationError",
    "HELDOUT_TEXT",
    "MARGIN_INSTALL",
    "Probe",
    "battery_hit",
    "battery_summary",
    "build_battery",
    "calibrate_battery",
    "candidate_bank_hash",
    "code_margin",
    "load_or_build_battery",
    "load_or_build_battery_batched",
    "logprob_sum",
    "nll_from_logits",
    "perplexity",
    "score_battery",
    "score_battery_batched",
    "validate_battery_candidates",
)


def test_probes_common_reexports_locality_surface():
    assert all(getattr(probes_common, name) is getattr(locality, name) for name in PUBLIC_NAMES)
