"""Tests for benchmark functions and their analytic gradients."""

import numpy as np
import pytest

from pbit.bench.functions import Ackley, Rastrigin, Rosenbrock, Sphere, get_function


def test_known_minima():
    assert Rastrigin().evaluate(np.zeros(2)) == pytest.approx(0.0, abs=1e-9)
    assert Ackley().evaluate(np.zeros(2)) == pytest.approx(0.0, abs=1e-9)
    assert Rosenbrock().evaluate(np.ones(2)) == pytest.approx(0.0, abs=1e-9)
    assert Sphere().evaluate(np.zeros(2)) == pytest.approx(0.0, abs=1e-9)
    assert Sphere().evaluate(np.zeros(5)) == pytest.approx(0.0, abs=1e-9)


def test_sphere_gradients_never_hit_the_runner_clip():
    """Sphere gradients fit inside the runner's clip on the whole init box.

    This is the property the quantization retest depends on: with
    ``|g| <= 4 < 5 = clip``, clipping is the identity and quantized
    gradients survive it. Rastrigin gradients (up to ~71) do not fit,
    which is why claim C1 was untestable there.
    """
    from pbit.bench.experiment import DEFAULT_CLIP

    fn = Sphere(dim=2)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        x = rng.uniform(*fn.init_bounds, size=fn.dim)
        worst = max(worst, float(np.abs(fn.grad(x)).max()))
    assert worst < DEFAULT_CLIP, f"max |g| = {worst} hits the clip"

    # And mid-trajectory points stay inside too (trajectories shrink inward).
    worst_traj = 0.0
    for _ in range(200):
        x = rng.uniform(-1.0, 1.0, size=fn.dim)
        worst_traj = max(worst_traj, float(np.abs(fn.grad(x)).max()))
    assert worst_traj < DEFAULT_CLIP


def test_gradients_match_finite_difference():
    rng = np.random.default_rng(0)
    for fn in [Ackley(), Rastrigin(), Rosenbrock(), Sphere()]:
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
    assert get_function("Sphere", dim=4).name == "Sphere"
    with pytest.raises(KeyError):
        get_function("Nope")


def test_dim_argument_minimas():
    assert np.allclose(Rastrigin(dim=4).argmin, np.zeros(4))
    assert np.allclose(Rosenbrock(dim=3).argmin, np.ones(3))
    assert np.allclose(Sphere(dim=5).argmin, np.zeros(5))


def test_sphere_converges_with_tuned_like_configs():
    """Sphere is solvable by every optimizer family in the study.

    Guards the quantization retest: if an optimizer cannot solve the
    clean cell, its noisy-clean shift measures failure, not noise.
    Uses small learning rates of the scale tuning selects on Sphere.
    """
    from pbit.bench import Experiment, QuantizeNoise
    from pbit.bench.specs import OptimizerSpec
    from pbit.optim import SGD, Adam, PBitOptimizer, SignSGD

    specs = [
        OptimizerSpec("pbit", lambda: PBitOptimizer(lr=0.05, tau=300)),
        OptimizerSpec("adam", lambda: Adam(lr=0.05)),
        OptimizerSpec("sgd", lambda: SGD(lr=0.05)),
        OptimizerSpec("signsgd", lambda: SignSGD(lr=0.05)),
    ]
    rep = Experiment(
        {
            "functions": [Sphere(dim=2)],
            "noises": [None, QuantizeNoise(bits=4, stochastic=True)],
            "optimizers": specs,
            "max_iter": 300,
            "n_runs": 3,
            "seed": 0,
        }
    ).run()
    for opt in ("pbit", "adam", "sgd", "signsgd"):
        cell = rep.get_cell("Sphere", "clean", opt)
        assert cell is not None
        # Clean Sphere from U(-2,2) starts at mean loss ~5.3; anything
        # under 1.0 after 300 steps has genuinely converged inward.
        assert cell.final_current_loss_mean < 1.0, (opt, cell.final_current_loss_mean)
