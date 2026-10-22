"""Significance testing over paired per-query scores."""

from __future__ import annotations

import numpy as np


def randomization_test(a, b, trials: int = 10000, seed: int = 0) -> float:
    """Two-sided paired (sign-flip) randomization test on the mean difference (Smucker 2007)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(trials, diff.size))
    null = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (trials + 1))


def bootstrap_ci(values, trials: int = 10000, seed: int = 0, level: float = 0.95):
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(trials, data.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values (family-wise error control)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, adjusted = len(items), 0.0, {}
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def auroc(positive_scores, negative_scores) -> float:
    """Probability a random positive outscores a random negative (ties count half)."""
    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)
    if not pos.size or not neg.size:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))
