"""Core p-bit mathematical primitives.

These are the building blocks for the p-bit optimizer and diagnostics:

- ``sigmoid`` : numerically stable sigmoid (no overflow at extreme inputs).
- ``sigmoid_probability`` : Boltzmann probability that a p-bit relaxes to +1.
- ``bernoulli_bit`` : sample p-bit states ``{+1, -1}`` with given probabilities.
- ``entropy`` : Shannon entropy of a per-coordinate flip-probability vector.
"""

from __future__ import annotations

import numpy as np

# Prefer scipy's stable expit if available; otherwise a manual branch whose
# recurrence keeps exp underflow/overflow at bay.
try:  # pragma: no cover - scipy is an optional convenience
    from scipy.special import expit as _expit

    def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
        return _expit(x).astype(np.float64, copy=False)

except ImportError:  # pragma: no cover - manual fallback

    def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        e = np.exp(-np.abs(x))
        out[pos] = 1.0 / (1.0 + e[pos])
        out[~pos] = e[~pos] / (1.0 + e[~pos])
        return out


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic sigmoid ``1 / (1 + exp(-x))``.

    Returns a scalar for scalar input, else an array. Never overflows/underflows
    to inf/nan for finite input.
    """
    arr = np.asarray(x, dtype=np.float64)
    scalar = arr.ndim == 0
    out = _stable_sigmoid(np.atleast_1d(arr))
    return out[0] if scalar else out


def sigmoid_probability(beta: float, grad: np.ndarray, g_scale: float) -> np.ndarray:
    """Boltzmann probability that a p-bit flips to ``+1``.

    Implements the 'legacy' convention ``P(σ=+1) = sigmoid(-beta * g / g_scale)``
    for the (normalized) gradient ``g``. ``g_scale`` is a positive normalization
    (typically ``mean(|g|) + eps``) that keeps the argument within a range where
    ``beta`` is meaningful.
    """
    arg = -np.asarray(beta, dtype=np.float64) * np.asarray(grad, dtype=np.float64) / float(g_scale)
    return _stable_sigmoid(arg)


def bernoulli_bit(rng: np.random.Generator, prob: np.ndarray) -> np.ndarray:
    """Sample p-bit states ``{+1, -1}`` where ``P(+1) = prob`` per coordinate."""
    p = np.asarray(prob, dtype=np.float64)
    return np.where(rng.random(p.shape) < p, 1.0, -1.0)


def entropy(prob: np.ndarray) -> float:
    """Shannon entropy (in nats) of a flip-probability vector.

    ``H = -sum(p log p + (1-p) log(1-p))``. Used as a thermodynamic diagnostic:
    values near ``log(2)`` mean the coordinate is fully stochastic; values near 0
    mean it has become deterministic.
    """
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(-np.sum(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))
