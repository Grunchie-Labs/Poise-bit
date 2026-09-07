"""Hashable, serializable specs for benchmarks.

The benchmark runner consumes immutable spec objects (`FunctionSpec`,
`NoiseSpec`, `OptimizerSpec`) so that the exact experiment configuration can be
hased into a reproducible ``config_hash`` and recorded in result metadata.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FunctionSpec:
    """A benchmark objective function.

    ``evaluate(x)`` must be a pure, deterministic function of ``x`` so the same
    config yields identical histories.
    """

    name: str
    dim: int
    init_bounds: tuple[float, float]
    clip_bounds: tuple[float, float]
    evaluate: Callable[[Any], float]
    grad: Callable[[Any], Any] | None = None
    minimum: float | None = None
    argmin: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "init_bounds": list(self.init_bounds),
            "clip_bounds": list(self.clip_bounds),
            "minimum": self.minimum,
        }


@dataclass(frozen=True)
class NoiseSpec:
    """A gradient noise transform. ``apply(g, rng) -> g_noisy``."""

    name: str
    apply: Callable[[Any, Any], Any]
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}


@dataclass(frozen=True)
class OptimizerSpec:
    """An optimizer factory plus its name and declared proxy FLOP cost.

    ``factory`` must return a fresh, resettable optimizer instance. ``flops_per_step``
    is a **proxy** (declared) cost, explicitly labeled, never merged with measured
    wall-time.
    """

    name: str
    factory: Callable[[], Any]
    flops_per_step: Callable[[int], int] | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}


def config_hash(*spec_lists: list[Any], seed: int, max_iter: int, n_runs: int, clip: float) -> str:
    """Stable SHA256 hash of the reproducible experiment configuration.

    ``None`` entries (used to mean "clean" noise) hash to a placeholder text.
    """
    payload: dict[str, Any] = {
        "seed": seed,
        "max_iter": max_iter,
        "n_runs": n_runs,
        "clip": clip,
        "functions": [s.to_dict() for s in spec_lists[0]],
        "noises": [s.to_dict() if s is not None else {"name": "clean", "params": {}} for s in spec_lists[1]],
        "optimizers": [s.to_dict() for s in spec_lists[2]],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
