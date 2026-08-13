"""Compatibility wrapper for :mod:`experiments.erasure.pilot`."""

from experiments.erasure.pilot import (
    CLIP_QUANTILE,
    FAMILIES,
    HEADER,
    MIN_AUC,
    QUANTILES,
    PilotCapture,
    PilotDream,
    _mean,
    auc,
    cli_main,
    eligible_positions,
    format_row,
    main,
    oracle_positions,
    quantile,
    readout_removals,
    recommend,
    scheme_weights,
    score_scheme,
    separability,
)

__all__ = ["PilotDream","PilotCapture","QUANTILES","FAMILIES","MIN_AUC","CLIP_QUANTILE","auc","eligible_positions","oracle_positions","separability","scheme_weights","quantile","readout_removals","_mean","score_scheme","recommend","HEADER","format_row","main"]


if __name__ == "__main__":
    cli_main()
