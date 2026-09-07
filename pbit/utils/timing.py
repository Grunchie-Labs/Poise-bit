"""Timing / measurement helpers.

Wall-time is *measured* hardware performance; it is reported as a distribution
and never conflated with algorithmic/compute metrics.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class TimingStats:
    mean: float
    median: float
    std: float
    min: float
    max: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }


def summarize_times(times: Iterable[float]) -> TimingStats:
    import numpy as np

    a = np.asarray(list(times), dtype=np.float64)
    if a.size == 0:
        return TimingStats(0.0, 0.0, 0.0, 0.0, 0.0)
    return TimingStats(
        mean=float(a.mean()),
        median=float(np.median(a)),
        std=float(a.std()),
        min=float(a.min()),
        max=float(a.max()),
    )


class Timer:
    """Context-manager wall-clock timer."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
        return False
