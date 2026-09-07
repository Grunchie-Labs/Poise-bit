"""Tests for pbit.core primitives: probability, schedule, rng."""

import numpy as np
import pytest

from pbit.core import (
    bernoulli_bit,
    constructor_rng,
    derive_rng,
    derive_seed,
    entropy,
    linear_cooling,
    sigmoid,
    sigmoid_probability,
)


# ----------------------------------------------------------------- sigmoid
def test_sigmoid_no_overflow():
    assert sigmoid(np.array([1000.0])) == 1.0
    assert sigmoid(np.array([-1000.0])) == 0.0
    assert np.isfinite(sigmoid(np.array([1e6, -1e6]))).all()


def test_sigmoid_symmetry():
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    s = sigmoid(x)
    assert np.allclose(s + sigmoid(-x), 1.0, atol=1e-12)


def test_sigmoid_midpoint():
    assert sigmoid(0.0) == 0.5


def test_sigmoid_scalar():
    assert abs(float(sigmoid(0.0)) - 0.5) < 1e-12


# ------------------------------------------------- probability / sampling
def test_sigmoid_probability_monotonic_decreasing_in_grad():
    beta = 2.0
    g_scale = 1.0
    g = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    p = sigmoid_probability(beta, g, g_scale)
    assert np.all(np.diff(p) < 0)  # larger gradient -> lower P(+1)


def test_sigmoid_probability_scale_invariance():
    beta = 2.0
    g = np.array([0.5, -0.3])
    p1 = sigmoid_probability(beta, g, g_scale=1.0)
    p2 = sigmoid_probability(beta, g * 2.0, g_scale=2.0)
    assert np.allclose(p1, p2)


def test_bernoulli_bit_mean_matches_probability():
    rng = np.random.default_rng(0)
    p = np.array([0.8, 0.2, 0.5])
    bits = np.stack([bernoulli_bit(rng, p) for _ in range(100000)])
    emp = np.mean(bits == 1.0, axis=0)
    assert np.allclose(emp, p, atol=0.01)


def test_bernoulli_bit_values():
    rng = np.random.default_rng(0)
    bits = bernoulli_bit(rng, np.array([1.0, 0.0]))
    assert bits[0] == 1.0 and bits[1] == -1.0


# ------------------------------------------------------------ entropy
def test_entropy_max_at_half():
    assert abs(entropy(np.array([0.5, 0.5])) - np.log(2.0) * 2) < 1e-6


def test_entropy_zero_at_deterministic():
    assert abs(entropy(np.array([0.0, 1.0, 0.0]))) < 1e-9


# ------------------------------------------------------------------- rng
def test_derive_seed_deterministic():
    assert derive_seed(42, "a", "b") == derive_seed(42, "a", "b")


def test_derive_seed_independent_tags():
    s1 = derive_seed(42, "a")
    s2 = derive_seed(42, "b")
    assert s1 != s2


def test_derive_rng_reproducible():
    r1 = derive_rng(42, "opt", "pbit")
    r2 = derive_rng(42, "opt", "pbit")
    assert np.array_equal(r1.random(16), r2.random(16))


def test_derive_rng_independent_across_tags():
    r1 = derive_rng(42, "opt", "pbit")
    r2 = derive_rng(42, "opt", "adam")
    assert not np.array_equal(r1.random(16), r2.random(16))


def test_constructor_rng_none_entropy():
    r = constructor_rng(None)
    assert isinstance(r, np.random.Generator)


# --------------------------------------------------------------- schedule
def test_linear_cooling_bounds():
    s = linear_cooling(beta0=2.0, tau=150.0, beta_cap=50.0)
    assert s(0) == 2.0
    assert s(75) == pytest.approx(3.0)
    assert s(100_000) == 50.0  # capped


def test_schedule_monotonic_nondecreasing():
    s = linear_cooling(2.0, 150.0, 50.0)
    vals = [s(t) for t in range(0, 5000, 25)]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))
