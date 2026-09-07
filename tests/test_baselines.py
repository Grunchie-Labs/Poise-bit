"""Tests for baseline optimizers and ask/tell optimizers."""

import numpy as np
import pytest

from pbit.optim import (
    SGD,
    Adam,
    AdamW,
    EvolutionStrategy,
    Langevin,
    Lion,
    Momentum,
    RMSProp,
    SignSGD,
    SimulatedAnnealing,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SGD(lr=0.1),
        lambda: Momentum(lr=0.05),
        lambda: Adam(lr=0.1),
        lambda: AdamW(lr=0.1),
        lambda: RMSProp(lr=0.1),
        lambda: SignSGD(lr=0.05),
        lambda: Lion(lr=0.01),
        lambda: Langevin(lr=0.05),
    ],
)
def test_baselines_converge_on_quadratic(factory):
    def f(x):
        return float(np.sum(x**2))

    def g(x):
        return 2 * x

    opt = factory()
    opt.reset(np.random.default_rng(0))
    x = np.array([1.0, -1.0, 0.5])
    best = float("inf")
    for t in range(1000):
        x = opt.step(x, g(x), t)
        best = min(best, f(x))
    assert best < 0.1


def test_no_state_leakage_after_reset():
    opt = Adam(lr=0.05)
    opt.reset(np.random.default_rng(0))
    opt.step(np.array([1.0, 2.0]), np.array([0.5, -0.5]), 0)
    # After a fresh reset, first-step momentum is zero, so the step uses only g.
    opt.reset(np.random.default_rng(1))
    x = np.array([1.0])
    g = np.array([2.0])
    out = opt.step(x, g, 0)
    assert np.allclose(out, np.array([1.0 - 0.05 * 2.0 / (np.sqrt((2.0) ** 2) + 1e-8)]))


def test_adam_first_step_math():
    rng = np.random.default_rng(0)
    opt = Adam(lr=0.1, b1=0.9, b2=0.999, eps=1e-8)
    opt.reset(rng)
    x = np.array([1.0])
    g = np.array([2.0])
    out = opt.step(x, g, 0)
    # t=1: m=g=2, v=4; mh=2, vh=4; step=0.1*2/sqrt(4)=0.1
    assert np.allclose(out, np.array([1.0 - 0.1]))


def test_simulated_annealing_runs():
    sa = SimulatedAnnealing(x0=np.array([1.0, -1.0]), step_scale=0.1, seed=0)
    sa.reset(np.random.default_rng(0))
    for _ in range(100):
        cand = sa.ask()
        sa.tell(float(np.sum(cand**2)))
    assert sa.done(100).shape == (2,)


def test_evolution_strategy_runs():
    es = EvolutionStrategy(x0=np.array([1.0, -1.0]), sigma=0.3, lam=10, seed=0)
    es.reset(np.random.default_rng(0))
    for _ in range(100):
        cand = es.ask()
        es.tell(float(np.sum(cand**2)))
    assert es.done(100).shape == (2,)
