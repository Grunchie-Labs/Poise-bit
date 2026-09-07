"""Optimizer protocols.

Two protocols are exposed so both gradient-style optimizers (PBit, Adam, SGD,
SignSGD, Lion, ...) and ask/tell optimizers (Simulated Annealing, Evolution
Strategy) share one interface without the ugly ``set_fns`` hack of the original.

- ``GradientOptimizer`` : ``step(x, grad, t) -> x_next``.
- ``AskTellOptimizer``  : ``ask() -> candidate`` / ``tell(fitness)`` /
  ``done(t) -> x``; used with a ``Fitness`` oracle by ``pbit.bench``.
- ``seedable(opt)``     : duck-typed resettable-with-rng contract. The benchmark
  always injects randomness through ``reset(rng)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GradientOptimizer(Protocol):
    """Optimizer driven by explicit gradient steps.

    Implementations must be resettable via ``reset(rng)`` so the benchmark can
    inject per-run RNG and guarantee no state leakage between runs.
    """

    def reset(self, rng: np.random.Generator) -> None: ...

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray: ...


@runtime_checkable
class AskTellOptimizer(Protocol):
    """Optimizer driven by a fitness oracle (ask/tell)."""

    def reset(self, rng: np.random.Generator) -> None: ...

    def ask(self) -> np.ndarray: ...

    def tell(self, fitness: float) -> None: ...

    def done(self, t: int) -> np.ndarray: ...
