"""Inverse-temperature (beta) schedules for the p-bit optimizer.

The p-bit optimizer anneals by increasing inverse temperature ``beta`` over
time, exactly like the original PBit: ``beta(t) = min(beta0 * (1 + t/tau),
beta_cap)``. The schedule is a callable ``schedule(t) -> float`` so users can
plug in custom schedules.
"""

from __future__ import annotations

from typing import Protocol


class Schedule(Protocol):
    def __call__(self, t: int) -> float: ...


def linear_cooling(beta0: float = 2.0, tau: float = 150.0, beta_cap: float = 50.0) -> Schedule:
    """Linear inverse-temperature schedule, capped.

    ``beta(t) = min(beta0 * (1 + t/tau), beta_cap)``.
    This is the default/legacy PBit schedule: it cools (increases ``beta``)
    so exploration-dominated early steps become exploitation-dominated later.
    """

    def schedule(t: int) -> float:
        return min(beta0 * (1.0 + float(t) / float(tau)), beta_cap)

    return schedule


def constant(beta: float = 1.0) -> Schedule:
    """Constant inverse temperature."""

    def schedule(t: int) -> float:
        return beta

    return schedule
