"""Tests for benchmark functions and their analytic gradients."""

import numpy as np
import pytest

from pbit.bench.functions import Ackley, Rastrigin, Rosenbrock, get_function


def test_known_minima():
    assert Rastrigin().evaluate(np.zeros(2)) == pytest.approx(0.0, abs=1e-9)
    assert Ackley().evaluate(np.zeros(2)) == pytest.approx(0.0, abs=1e-9)
    assert Rosenbrock().evaluate(np.ones(2)) == pytest.approx(0.0, abs=1e-9)


def test_gradients_match_finite_difference():
    rng = np.random.default_rng(0)
    for fn in [Ackley(), Rastrigin(), Rosenbrock()]:
        eps = 1e-7 if fn.name == "Ackley" else 1e-6
        for _ in range(10):
            x = rng.uniform(-2, 2, size=fn.dim)
            g = fn.grad(x)
            fd = np.array(
                [
                    (fn.evaluate(x + np.eye(fn.dim)[i] * eps) - fn.evaluate(x - np.eye(fn.dim)[i] * eps))
                    / (2 * eps)
                    for i in range(fn.dim)
                ]
            )
            assert np.allclose(g, fd, atol=1e-5), f"{fn.name}: {g} vs {fd}"


def test_domains():
    assert Rastrigin().init_bounds == (-5.12, 5.12)
    assert Rosenbrock().init_bounds == (-2.0, 2.0)
    assert Ackley().clip_bounds == (-32.768, 32.768)


def test_get_function():
    assert get_function("Ackley", dim=3).dim == 3
    with pytest.raises(KeyError):
        get_function("Nope")


def test_dim_argument_minimas():
    assert np.allclose(Rastrigin(dim=4).argmin, np.zeros(4))
    assert np.allclose(Rosenbrock(dim=3).argmin, np.ones(3))
