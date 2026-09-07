"""Tests for PBitOptimizer: exact v2 math, step-size modes, determinism."""

import numpy as np
import pytest

from pbit.optim import PBitOptimizer


def _set_all(a, b):
    return np.allclose(a, b, atol=1e-12)


def test_v2_exact_update_rule():
    """Proportional (v2) rule: x' = x + lr*(|g|+eps)*sigma with P(+1)=sigmoid(-beta*g/g_scale)."""
    from pbit.core import sigmoid

    rng = np.random.default_rng(1234)
    opt = PBitOptimizer(lr=0.01, beta0=2.0, tau=150.0)
    opt.reset(rng)
    x = np.array([1.0, -1.0])
    g = np.array([0.4, -0.2])
    t = 0

    beta = opt.beta(t)
    g_scale = float(np.mean(np.abs(g))) + 1e-8
    prob = sigmoid(-beta * g / g_scale)
    expected_sigma = np.where(rng.random(2) < prob, 1.0, -1.0)
    expected = x + 0.01 * (np.abs(g) + 1e-8) * expected_sigma

    out = opt.step(x, g, t)
    assert _set_all(out, expected)
    # Diagnostics
    assert opt.state.beta == pytest.approx(beta)
    assert opt.state.g_scale == pytest.approx(g_scale)
    assert np.allclose(opt.state.prob, prob)


def test_constant_step_mode():
    rng = np.random.default_rng(9)
    opt = PBitOptimizer(step_size="constant", lr=0.1)
    opt.reset(rng)
    x = np.zeros(3)
    g = np.array([1e6, 1e6, -1e6])  # direction decided by sigma
    out = opt.step(x, g, 0)
    step = out - x
    assert np.allclose(np.abs(step), 0.1)  # fixed magnitude


def test_floor_step_mode_escapes_zero():
    rng = np.random.default_rng(7)
    # proportional has ~zero step on zero gradient; floor has floor*|sigma|
    opt_floor = PBitOptimizer(step_size="floor", floor=1e-3, lr=0.1)
    opt_floor.reset(rng)
    x = np.array([0.0, 0.0])
    g = np.array([1e-12, 1e-12])
    out = opt_floor.step(x, g, 0)
    assert float(np.abs(out[0])) > 1e-6  # moved despite tiny gradient


def test_step_size_validation():
    with pytest.raises(ValueError):
        PBitOptimizer(step_size="bogus")


def test_seeded_reproducibility():
    a1 = PBitOptimizer(seed=42)
    a2 = PBitOptimizer(seed=42)
    r1 = PBitOptimizer(seed=43)
    x = np.zeros(4)
    g = np.ones(4)
    o1 = a1.step(x.copy(), g.copy(), 3)
    o2 = a2.step(x.copy(), g.copy(), 3)
    o3 = r1.step(x.copy(), g.copy(), 3)
    assert _set_all(o1, o2)
    assert not _set_all(o1, o3)


def test_reset_injects_rng():
    opt = PBitOptimizer(seed=1)
    opt.reset(np.random.default_rng(99))
    out1 = opt.step(np.zeros(4), np.ones(4), 0)
    opt.reset(np.random.default_rng(99))
    out2 = opt.step(np.zeros(4), np.ones(4), 0)
    assert _set_all(out1, out2)


def test_dim1():
    opt = PBitOptimizer(seed=0)
    opt.reset(np.random.default_rng(0))
    out = opt.step(np.array([0.0]), np.array([1.0]), 0)
    assert out.shape == (1,)


def test_large_beta_stable():
    opt = PBitOptimizer(beta0=1e6, tau=1.0, beta_cap=1e12)
    opt.reset(np.random.default_rng(0))
    # Extreme beta -> prob saturates; sigmoid must not NaN.
    out = opt.step(np.zeros(3), np.ones(3) * 1e-3, 0)
    assert np.isfinite(out).all()


def test_converges_on_rosenbrock():
    """PBit reaches the Rosenbrock valley from standard random starts.

    Use multiple seeds from U(-2,2) (like the benchmark runner) and clip
    gradients, matching real usage. PBit is stochastic, so assert on the median
    best-so-far across seeds rather than any single trajectory. A longer cooling
    timescale (tau) keeps exploration up long enough to find the valley.
    """
    from pbit.bench.functions import Rosenbrock

    fn = Rosenbrock(dim=2)
    bests = []
    for s in range(20):
        opt = PBitOptimizer(lr=0.05, tau=300)
        opt.reset(np.random.default_rng(s))
        x = np.random.default_rng(1000 + s).uniform(-2.0, 2.0, size=2)
        b = float("inf")
        for t in range(1500):
            g = np.clip(fn.grad(x), -5.0, 5.0)
            x = opt.step(x, g, t)
            b = min(b, float(fn.evaluate(x)))
        bests.append(b)
    assert np.median(bests) < 1.0


def test_params_snapshot():
    opt = PBitOptimizer(lr=0.01, step_size="floor", floor=1e-3)
    p = opt.params()
    assert p["step_size"] == "floor"
    assert p["floor"] == 1e-3


def test_flip_probability_by_coordinate():
    opt = PBitOptimizer(seed=1)
    opt.reset(np.random.default_rng(1))
    opt.step(np.zeros(3), np.ones(3), 0)
    assert opt.flip_probability_by_coordinate().shape == (3,)
