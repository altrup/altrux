"""Diagnostic: lag-1 cosine between consecutive Adam steps on independent gradients.

Usage: python -m experiments.lama_ckl.diagnostics.adam_momentum_null [d] [steps]
The null for `census.json`'s cos_lag1: what momentum alone (beta1 0.9, beta2 0.999,
bias-corrected) produces when every cycle's gradient is unrelated to the last.
"""

import sys

import numpy as np

from progress import ts


def main() -> None:
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rng = np.random.default_rng(0)
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros(d)
    v = np.zeros(d)
    prev = None
    for t in range(1, steps + 1):
        g = rng.standard_normal(d)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        step = (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        if prev is not None:
            cos = float(step @ prev / (np.linalg.norm(step) * np.linalg.norm(prev)))
            print(f"[{ts()}] steps {t - 1}->{t}: cos_lag1 {cos:.3f}")
        prev = step


if __name__ == "__main__":
    main()
