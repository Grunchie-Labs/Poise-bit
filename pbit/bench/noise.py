"""Gradient noise suite: AI/hardware-relevant distortions.

These mirror the environment a p-bit / low-precision optimizer must survive:

- ``NoNoise``        : identity (clean gradients).
- ``GaussianNoise``  : additive gaussian noise.
- ``CorruptionNoise``: randomly flip some components and amplify them.
- ``QuantizeNoise``  : uniform quantization to ``bits``; optionally *stochastic*
  (randomized rounding), which is closer to real low-precision hardware.
- ``SignNoise``      : only the sign of each gradient component is kept (1-bit).
- ``ClipNoise``      : clip the gradient norm.

Every transform takes an explicit ``rng`` so the benchmark can inject the
shared ``noise`` stream for comparable noisy gradients across optimizers.
"""

from __future__ import annotations

import numpy as np

from pbit.bench.specs import NoiseSpec


def _quantize(g: np.ndarray, bits: int, stochastic: bool, rng: np.random.Generator) -> np.ndarray:
    g = np.asarray(g, dtype=np.float64)
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits!r}")
    levels = float(2**bits - 1)
    gmin, gmax = g.min(), g.max()
    if gmax == gmin:
        return g
    g_norm = (g - gmin) / (gmax - gmin)
    scaled = g_norm * levels
    if stochastic:
        # Randomized rounding: floor with probability proportional to the
        # fractional part (unbiased) -- models real stochastic low-precision HW.
        floor = np.floor(scaled)
        frac = scaled - floor
        rounded = floor + (rng.random(g.shape) < frac).astype(np.float64)
    else:
        rounded = np.round(scaled)
    return (rounded / levels) * (gmax - gmin) + gmin


def _clip_norm(g: np.ndarray, max_norm: float) -> np.ndarray:
    g = np.asarray(g, dtype=np.float64)
    norm = np.linalg.norm(g)
    if norm > max_norm:
        return g * (max_norm / norm)
    return g


def NoNoise() -> NoiseSpec:
    def apply(g, rng):
        return np.asarray(g, dtype=np.float64)

    return NoiseSpec(name="clean", apply=apply)


def GaussianNoise(sigma: float = 0.5) -> NoiseSpec:
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma!r}")
    def apply(g, rng):
        return np.asarray(g, dtype=np.float64) + rng.normal(0.0, sigma, np.asarray(g).shape)

    return NoiseSpec(name=f"GaussianNoise_sigma={sigma}", apply=apply, params={"sigma": sigma})


def CorruptionNoise(p: float = 0.2, amplify: float = 3.0) -> NoiseSpec:
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if amplify < 0:
        raise ValueError(f"amplify must be non-negative, got {amplify!r}")
    def apply(g, rng):
        g = np.asarray(g, dtype=np.float64)
        mask = rng.random(g.shape) < p
        gc = g.copy()
        gc[mask] = -gc[mask] * amplify
        return gc

    return NoiseSpec(name=f"CorruptionNoise_p={p}", apply=apply, params={"p": p, "amplify": amplify})


def QuantizeNoise(bits: int = 4, stochastic: bool = False) -> NoiseSpec:
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits!r}")
    def apply(g, rng):
        return _quantize(g, bits, stochastic, rng)

    return NoiseSpec(
        name=f"QuantizeNoise_bits={bits}_stochastic={stochastic}",
        apply=apply,
        params={"bits": bits, "stochastic": stochastic},
    )


def SignNoise() -> NoiseSpec:
    def apply(g, rng):
        return np.sign(np.asarray(g, dtype=np.float64))

    return NoiseSpec(name="SignNoise", apply=apply)


def ClipNoise(max_norm: float = 5.0) -> NoiseSpec:
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm!r}")
    def apply(g, rng):
        return _clip_norm(g, max_norm)

    return NoiseSpec(name=f"ClipNoise_max_norm={max_norm}", apply=apply, params={"max_norm": max_norm})
