"""Tests for noise injectors."""

import numpy as np

from pbit.bench import (
    ClipNoise,
    CorruptionNoise,
    GaussianNoise,
    NoNoise,
    QuantizeNoise,
    SignNoise,
)


def test_nonoise_identity():
    rng = np.random.default_rng(0)
    g = np.array([0.1, -0.2, 0.3])
    assert np.allclose(NoNoise().apply(g, rng), g)


def test_gaussian_stats():
    rng = np.random.default_rng(0)
    g = np.zeros(20000)
    out = GaussianNoise(sigma=0.5).apply(g, rng)
    assert abs(np.std(out) - 0.5) < 0.02


def test_corruption_flips_some():
    rng = np.random.default_rng(0)
    g = np.ones(10000)
    out = CorruptionNoise(p=0.2).apply(g, rng)
    flipped = out < 0
    assert 0.15 < flipped.mean() < 0.25  # ~20% flipped and amplified
    assert np.all(np.abs(np.abs(out)[flipped]) > 1)


def test_quantize_deterministic():
    rng = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    g = np.linspace(-1, 1, 100)
    q1 = QuantizeNoise(bits=4, stochastic=False).apply(g, rng)
    q2 = QuantizeNoise(bits=4, stochastic=False).apply(g, rng2)
    assert np.array_equal(q1, q2)  # deterministic quantizer ignores rng


def test_quantize_levels():
    rng = np.random.default_rng(0)
    g = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    out = QuantizeNoise(bits=3, stochastic=False).apply(g, rng)
    # 3 bits -> 2**3-1 = 7 intervals, levels spaced by (max-min)/7
    levels = g.min() + np.arange(2**3) / (2**3 - 1) * (g.max() - g.min())
    for v in out:
        assert any(np.isclose(v, L, atol=1e-9) for L in levels)


def test_quantize_stochastic_seeded():
    rng = np.random.default_rng(3)
    rng2 = np.random.default_rng(3)
    g = np.linspace(-1, 1, 50)
    a = QuantizeNoise(bits=2, stochastic=True).apply(g, rng)
    b = QuantizeNoise(bits=2, stochastic=True).apply(g, rng2)
    assert np.array_equal(a, b)
    # stochastic rounding stays within neighboring deterministic levels
    assert np.all(a >= g.min() - 1e-9)
    assert np.all(a <= g.max() + 1e-9)


def test_sign_noise():
    rng = np.random.default_rng(0)
    g = np.array([1.5, -2.0, 0.0])
    out = SignNoise().apply(g, rng)
    assert np.array_equal(out, np.array([1.0, -1.0, 0.0]))


def test_clip_noise():
    rng = np.random.default_rng(0)
    g = np.array([10.0, 0.0, 0.0])
    out = ClipNoise(max_norm=5.0).apply(g, rng)
    assert abs(np.linalg.norm(out) - 5.0) < 1e-9
