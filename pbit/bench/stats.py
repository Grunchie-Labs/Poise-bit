"""Statistics for comparing stochastic optimizers.

The benchmark draws start positions and noise from streams shared across
optimizers within a run (common random numbers). Two optimizers in the same
(function, noise, run) cell therefore start at the same point and see the same
gradient perturbations, so their per-run terminal losses form **paired**
observations. Every comparison here is paired: differencing within a run
cancels the shared start position and shared noise, which removes most of the
between-run variance that made unpaired means useless on these landscapes.

Reported quantities:

- ``paired_diff``   : mean within-run difference (a - b)
- ``bootstrap_ci``  : percentile bootstrap CI over runs
- ``sign_flip_p``   : two-sided exact randomization test on within-run signs
- ``holm_bonferroni``: multiplicity correction across a family of comparisons

The sign-flip test is used instead of a t-test because per-run loss
differences on Rastrigin and Rosenbrock are heavy-tailed and the sample sizes
here are small. The randomization null is exact and assumes only exchangeable
signs under the null of no effect.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Comparison:
    """Result of one paired comparison between two conditions."""

    label: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_lo: float
    ci_hi: float
    p_value: float
    sd_diff: float

    @property
    def significant(self) -> bool:
        """True when the 95% CI excludes zero (two-sided, alpha=0.05)."""
        return (self.ci_lo > 0.0) or (self.ci_hi < 0.0)

    @property
    def direction(self) -> str:
        if not self.significant:
            return "no detectable difference"
        return "a better" if self.mean_diff < 0 else "b better"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n": self.n,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "mean_diff": self.mean_diff,
            "sd_diff": self.sd_diff,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "p_value": self.p_value,
            "significant": self.significant,
            "direction": self.direction,
        }


def bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 20000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def sign_flip_p(diffs: np.ndarray, n_perm: int = 20000, seed: int = 6789) -> float:
    """Two-sided randomization test by flipping signs of paired differences.

    Null: the sign of each within-run difference is equally likely, i.e. the
    conditions are exchangeable within a run. Exact under that null; the
    Monte-Carlo p-value carries a resolution of about ``1 / (n_perm + 1)``.
    """
    d = np.asarray(diffs, dtype=np.float64)
    d = d[d != 0.0]
    if d.size == 0:
        return 1.0
    observed = abs(float(np.mean(d)))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    null_means = np.abs((signs * d).mean(axis=1))
    # +1 keeps the test conservative (never understates p).
    return float((np.sum(null_means >= observed) + 1) / (n_perm + 1))


def paired_compare(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    label: str = "",
) -> Comparison:
    """Compare two paired per-run series on the mean of ``a - b``."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError(f"paired series must match in shape: {aa.shape} vs {bb.shape}")
    d = aa - bb
    lo, hi = bootstrap_ci(d)
    return Comparison(
        label=label,
        n=int(d.size),
        mean_a=float(np.mean(aa)),
        mean_b=float(np.mean(bb)),
        mean_diff=float(np.mean(d)),
        ci_lo=lo,
        ci_hi=hi,
        p_value=sign_flip_p(d),
        sd_diff=float(np.std(d, ddof=1)) if d.size > 1 else float("nan"),
    )


def holm_bonferroni(comparisons: Sequence[Comparison], alpha: float = 0.05) -> list[Comparison]:
    """Adjust p-values across a family of comparisons, keeping the order.

    Returns new ``Comparison`` objects with Holm-adjusted p-values. Holm is used
    rather than Bonferroni because it is uniformly more powerful and still
    controls the family-wise error rate.
    """
    items = list(comparisons)
    m = len(items)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: items[i].p_value)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * items[i].p_value
        running = max(running, min(val, 1.0))
        adjusted[i] = running
    out: list[Comparison] = []
    for i, c in enumerate(items):
        out.append(
            Comparison(
                label=c.label,
                n=c.n,
                mean_a=c.mean_a,
                mean_b=c.mean_b,
                mean_diff=c.mean_diff,
                ci_lo=c.ci_lo,
                ci_hi=c.ci_hi,
                p_value=adjusted[i],
                sd_diff=c.sd_diff,
            )
        )
    return out


def describe(values: Sequence[float] | np.ndarray, label: str = "") -> dict:
    """Mean, sd, and 95% CI of a single series, for honest reporting."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"label": label, "n": 0}
    lo, hi = bootstrap_ci(v)
    return {
        "label": label,
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)) if v.size > 1 else float("nan"),
        "median": float(np.median(v)),
        "ci_lo": lo,
        "ci_hi": hi,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }
