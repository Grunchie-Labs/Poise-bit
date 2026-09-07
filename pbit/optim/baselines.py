"""Gradient-style baseline optimizers (all implement ``GradientOptimizer``).

These are the standard references used for comparison. Each is a small,
self-contained NumPy implementation with a uniform ``reset(rng)`` /
``step(x, grad, t)`` interface. ``rng`` is accepted for interface uniformity
even where a baseline is deterministic (SGD, Adam, ...).

AI-relevant baselines included: AdamW (practical deep-learning default),
SignSGD (closest conceptual baseline to PBit), and Lion (modern sign/momentum).
"""

from __future__ import annotations

import numpy as np

from pbit.core.rng import constructor_rng

EPS = 1e-8


class _Base:
    """Shared reset plumbing for gradient optimizers."""

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = constructor_rng(seed)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng if rng is not None else constructor_rng(self._seed)
        self._state_reset()

    def _state_reset(self) -> None:  # pragma: no cover - overridden
        pass

    def params(self) -> dict:
        raise NotImplementedError


class SGD(_Base):
    def __init__(self, lr: float = 0.01, seed: int | None = None) -> None:
        super().__init__(seed)
        self.lr = float(lr)

    def _state_reset(self) -> None:
        pass

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        grad = np.asarray(grad, dtype=np.float64)
        if not np.all(np.isfinite(grad)):
            raise ValueError(
                "gradient must be finite; got nan/inf. "
                "Consider clipping or checking your loss function."
            )
        return x - self.lr * grad

    def params(self) -> dict:
        return {"lr": self.lr}


class Momentum(_Base):
    def __init__(self, lr: float = 0.01, mu: float = 0.9, seed: int | None = None) -> None:
        super().__init__(seed)
        self.lr = float(lr)
        self.mu = float(mu)
        self._v: np.ndarray | None = None

    def _state_reset(self) -> None:
        self._v = None

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        if self._v is None:
            self._v = np.zeros_like(x)
        self._v = self.mu * self._v - self.lr * grad
        return x + self._v

    def params(self) -> dict:
        return {"lr": self.lr, "mu": self.mu}


class Adam(_Base):
    def __init__(
        self,
        lr: float = 0.01,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = EPS,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        self.lr = float(lr)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.eps = float(eps)
        self._m: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._t = 0

    def _state_reset(self) -> None:
        self._m = None
        self._v = None
        self._t = 0

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        if self._m is None:
            self._m = np.zeros_like(x)
            self._v = np.zeros_like(x)
        self._t += 1
        b1, b2 = self.b1, self.b2
        self._m = b1 * self._m + (1 - b1) * grad
        self._v = b2 * self._v + (1 - b2) * grad**2
        mh = self._m / (1 - b1**self._t)
        vh = self._v / (1 - b2**self._t)
        return x - self.lr * mh / (np.sqrt(vh) + self.eps)

    def params(self) -> dict:
        return {"lr": self.lr, "b1": self.b1, "b2": self.b2, "eps": self.eps}


class AdamW(_Base):
    """Adam with decoupled weight decay (the practical deep-learning default)."""

    def __init__(
        self,
        lr: float = 0.01,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = EPS,
        weight_decay: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        self.lr = float(lr)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._m: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._t = 0

    def _state_reset(self) -> None:
        self._m = None
        self._v = None
        self._t = 0

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        if self._m is None:
            self._m = np.zeros_like(x)
            self._v = np.zeros_like(x)
        self._t += 1
        b1, b2 = self.b1, self.b2
        g = grad + self.weight_decay * x  # decoupled decay folded into gradient
        self._m = b1 * self._m + (1 - b1) * g
        self._v = b2 * self._v + (1 - b2) * g**2
        mh = self._m / (1 - b1**self._t)
        vh = self._v / (1 - b2**self._t)
        step = self.lr * mh / (np.sqrt(vh) + self.eps) + self.lr * self.weight_decay * x
        return x - step

    def params(self) -> dict:
        return {
            "lr": self.lr,
            "b1": self.b1,
            "b2": self.b2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        }


class RMSProp(_Base):
    def __init__(self, lr: float = 0.01, decay: float = 0.9, eps: float = EPS, seed: int | None = None) -> None:
        super().__init__(seed)
        self.lr = float(lr)
        self.decay = float(decay)
        self.eps = float(eps)
        self._ms: np.ndarray | None = None

    def _state_reset(self) -> None:
        self._ms = None

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        if self._ms is None:
            self._ms = np.zeros_like(x)
        self._ms = self.decay * self._ms + (1 - self.decay) * grad**2
        return x - self.lr * grad / (np.sqrt(self._ms) + self.eps)

    def params(self) -> dict:
        return {"lr": self.lr, "decay": self.decay, "eps": self.eps}


class SignSGD(_Base):
    """Binary-direction SGD: moves one fixed magnitude step in sign(grad).

    This is the closest conventional baseline to PBit (binary direction), but it
    is deterministic (no Boltzmann probability, no annealing).
    """

    def __init__(self, lr: float = 0.01, seed: int | None = None) -> None:
        super().__init__(seed)
        self.lr = float(lr)

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        return x - self.lr * np.sign(grad)

    def params(self) -> dict:
        return {"lr": self.lr}


class Lion(_Base):
    """Lion: sign(weighted interpolation of momenta); uses sign-preserving momentum."""

    def __init__(
        self,
        lr: float = 0.01,
        b1: float = 0.9,
        b2: float = 0.99,
        weight_decay: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        self.lr = float(lr)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.weight_decay = float(weight_decay)
        self._m: np.ndarray | None = None

    def _state_reset(self) -> None:
        self._m = None

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        if self._m is None:
            self._m = np.zeros_like(x)
        g = grad + self.weight_decay * x
        update = np.sign(self.b1 * self._m + (1 - self.b1) * g)
        self._m = self.b2 * self._m + (1 - self.b2) * g
        return x - self.lr * update

    def params(self) -> dict:
        return {"lr": self.lr, "b1": self.b1, "b2": self.b2, "weight_decay": self.weight_decay}


class Langevin(_Base):
    """Stochastic Gradient Langevin Dynamics: SGD + injected noise."""

    def __init__(self, lr: float = 0.01, temperature: float = 0.1, seed: int | None = None) -> None:
        super().__init__(seed)
        self.lr = float(lr)
        self.temperature = float(temperature)

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        noise = np.sqrt(2 * self.lr * self.temperature) * self._rng.standard_normal(x.shape)
        return x - self.lr * grad + noise

    def params(self) -> dict:
        return {"lr": self.lr, "temperature": self.temperature}
