"""The p-bit optimizer: a probabilistic binary-direction optimizer.

Core update rule (exact "v2" behaved rule):

    beta(t)  = schedule(t)                       # inverse temperature, rises over time
    g_scale  = mean(|grad|) + eps                # per-step normalization for beta sensitivity
    P(+1)    = sigmoid(-beta * grad / g_scale)   # Boltzmann flip probability (legacy convention)
    sigma    ~ Bernoulli(P(+1)) in {+1, -1}      # binary stochastic direction decision
    step     = step_size * sigma
    x       += step

The binary *direction* is a thermal (Boltzmann) decision while the *magnitude*
of the step is deterministic and gradient-proportional (``proportional`` mode),
so on tight multimodal basins the step can be small enough not to skip the
basin while direction noise provides exploration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from pbit.core.probability import bernoulli_bit, entropy, sigmoid
from pbit.core.rng import constructor_rng
from pbit.core.schedule import Schedule, linear_cooling

EPS = 1e-8

StepSizeMode = str


@dataclass
class PBitState:
    """Mutable state exposed to diagnostics / reporting after a step."""

    beta: float = 0.0
    g_scale: float = EPS
    prob: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    step_size: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    t: int = 0

    def mean_flip_probability(self) -> float:
        if self.prob.size == 0:
            return 0.0
        return float(np.mean(self.prob))

    def entropy(self) -> float:
        return float(entropy(self.prob)) if self.prob.size else 0.0


class PBitOptimizer:
    """Probabilistic binary-direction optimizer.

    Parameters
    ----------
    lr:
        Base learning rate (step magnitude scale).
    beta0:
        Initial inverse temperature.
    tau:
        Cooling timescale (how many steps for ``beta`` to double).
    beta_cap:
        Maximum ``beta`` (numerical safety / determinism at high anneal).
    step_size:
        One of ``"proportional"`` (default, v2), ``"constant"`` (v1 escape), or
        ``"floor"`` (proportional with a small magnitude floor).
    floor:
        Magnitude floor used only by ``step_size="floor"``.
    convention:
        ``"legacy"`` (sigmoid(-beta*grad/g_scale)). Reserved for alternative
        probability conventions in future work.
    seed:
        Optional seed for standalone use. The benchmark always calls
        ``reset(rng)`` instead, injecting a private per-run stream.
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta0: float = 2.0,
        tau: float = 150.0,
        beta_cap: float = 50.0,
        step_size: StepSizeMode = "proportional",
        floor: float = 0.0,
        convention: str = "legacy",
        seed: int | None = None,
    ) -> None:
        if step_size not in ("proportional", "constant", "floor"):
            raise ValueError(f"unknown step_size {step_size!r}")
        if convention != "legacy":
            raise ValueError(f"unknown convention {convention!r}")
        self.lr = float(lr)
        self.beta0 = float(beta0)
        self.tau = float(tau)
        self.beta_cap = float(beta_cap)
        self.step_size = step_size
        self.floor = float(floor)
        self.convention = convention
        self._seed = seed
        self._rng = constructor_rng(seed)
        self._schedule: Schedule = linear_cooling(beta0, tau, beta_cap)
        self.state = PBitState()

    # ------------------------------------------------------------------ RNG
    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Reset state and (optionally) inject a fresh RNG stream.

        ``rng=None`` restores the constructor-provided seed stream (or OS
        entropy if the constructor seed was ``None``). The benchmark always
        passes the private per-optimizer generator here.
        """
        self._rng = rng if rng is not None else constructor_rng(self._seed)
        self.state = PBitState()

    # ------------------------------------------------------------- public
    def beta(self, t: int) -> float:
        return float(self._schedule(t))

    def params(self) -> dict:
        """Hyperparameters snapshot (for reporting / reproducibility)."""
        return {
            "lr": self.lr,
            "beta0": self.beta0,
            "tau": self.tau,
            "beta_cap": self.beta_cap,
            "step_size": self.step_size,
            "floor": self.floor,
            "convention": self.convention,
        }

    def configure_schedule(self, schedule: Callable[[int], float]) -> None:
        """Override the temperature schedule with a custom callable."""
        self._schedule = schedule

    # ------------------------------------------------------------- core
    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        grad = np.asarray(grad, dtype=np.float64)

        beta = self.beta(t)
        g_scale = float(np.mean(np.abs(grad))) + EPS

        prob = sigmoid(-beta * grad / g_scale)
        sigma = bernoulli_bit(self._rng, prob)

        abs_g = np.abs(grad)
        if self.step_size == "constant":
            mag = np.full_like(abs_g, self.lr)
        elif self.step_size == "floor":
            mag = self.lr * (abs_g + self.floor + EPS)
        else:  # proportional (default)
            mag = self.lr * (abs_g + EPS)

        step = mag * sigma

        self.state = PBitState(
            beta=beta,
            g_scale=g_scale,
            prob=prob,
            step_size=step,
            t=t,
        )

        return x + step

    def state_dict(self) -> dict:
        """JSON-serializable diagnostics for this run (for reporting)."""
        return {
            "beta_t": self.state.beta,
            "g_scale_t": self.state.g_scale,
            "mean_flip_probability_t": self.state.mean_flip_probability(),
            "entropy_t": self.state.entropy(),
        }

    def flip_probability_by_coordinate(self) -> np.ndarray:
        return self.state.prob.copy()
