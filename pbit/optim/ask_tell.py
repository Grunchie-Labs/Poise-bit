"""Ask/tell optimizers (Simulated Annealing, Evolution Strategy).

These do not consume gradients directly; they operate through a fitness oracle
via ``ask()`` / ``tell(fitness)``. The benchmark adapter (`pbit.bench`) bridges
the ``GradientOptimizer`` runner to these by evaluating the fitness of the
candidate produced by ``ask()``.
"""

from __future__ import annotations

import numpy as np

from pbit.core.rng import constructor_rng


class SimulatedAnnealing:
    """Ask/tell simulated annealing over the current point.

    ``ask()`` returns a perturbed candidate around the current point; ``tell``
    accepts it (becoming the current point) with a Boltzmann acceptance
    probability that cools over time.
    """

    def __init__(
        self,
        x0: np.ndarray,
        step_scale: float = 0.1,
        t0: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.step_scale = float(step_scale)
        self.t0 = float(t0)
        self._seed = seed
        self.reset(constructor_rng(seed))

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng if rng is not None else constructor_rng(self._seed)
        self.x = self.x0.copy()
        self.x_best = self.x.copy()
        self.f_best = np.inf
        self.f_current = np.inf
        self.t = 0
        self._candidate = self.x.copy()

    def ask(self) -> np.ndarray:
        self._candidate = self.x + self.step_scale * self._rng.standard_normal(self.x.shape)
        return self._candidate

    def tell(self, fitness: float) -> None:
        # Acceptance: always accept improvements; accept worse with Boltzmann prob,
        # where the effective temperature cools over time.
        t_eff = self.t0 / (1.0 + self.t)
        delta = fitness - self.f_current
        accept = fitness < self.f_current or self._rng.random() < np.exp(-max(delta, 0.0) / max(t_eff, 1e-12))
        if accept:
            self.x = self._candidate.copy()
            self.f_current = fitness
        self.f_best = min(self.f_best, fitness)
        if fitness <= self.f_best:
            self.x_best = self._candidate.copy()
        self.t += 1

    def done(self, t: int) -> np.ndarray:
        return self.x_best

    def params(self) -> dict:
        return {"step_scale": self.step_scale, "t0": self.t0}


class EvolutionStrategy:
    """(mu, lambda) Evolution Strategy via ask/tell.

    Generates ``lambda`` perturbed candidates, evaluates their fitness through
    ``tell``, and updates the center towards the fittest direction.
    """

    def __init__(
        self,
        x0: np.ndarray,
        sigma: float = 0.3,
        lam: int = 10,
        seed: int | None = None,
    ) -> None:
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.sigma = float(sigma)
        self.lam = int(lam)
        self._seed = seed
        self.reset(constructor_rng(seed))

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng if rng is not None else constructor_rng(self._seed)
        self.x = self.x0.copy()
        self.x_best = self.x.copy()
        self.f_best = np.inf
        self._candidates: list[np.ndarray] = []
        self._fitness: list[float] = []
        self._gen = 0

    def ask(self) -> np.ndarray:
        if not self._candidates:
            self._candidates = [self.x + self.sigma * self._rng.standard_normal(self.x.shape) for _ in range(self.lam)]
        return self._candidates[len(self._fitness)]

    def tell(self, fitness: float) -> None:
        self._fitness.append(fitness)
        if fitness < self.f_best:
            self.f_best = fitness
            self.x_best = self.x.copy()
        if len(self._fitness) == self.lam:
            # Weighted average of perturbed candidates, inverse-fitness weighting.
            f = np.array(self._fitness)
            w = 1.0 / (np.abs(f - f.min()) + 1e-9)
            w = w / w.sum()
            cand = np.array(self._candidates)
            self.x = np.sum(cand * w[:, None], axis=0)
            self._candidates = []
            self._fitness = []
            self._gen += 1

    def done(self, t: int) -> np.ndarray:
        return self.x_best

    def params(self) -> dict:
        return {"sigma": self.sigma, "lam": self.lam}
