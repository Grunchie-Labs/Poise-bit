"""End-to-end tests for Experiment/Report, reproducibility, and export."""

import json
from pathlib import Path

import numpy as np
import pytest

from pbit.bench import (
    Experiment,
    GaussianNoise,
    QuantizeNoise,
    Rastrigin,
    Rosenbrock,
    compare_reports,
)
from pbit.bench.specs import OptimizerSpec
from pbit.optim import Adam, Langevin, PBitOptimizer


def _specs():
    def fp(dim):
        return dim * 6

    return [
        OptimizerSpec("pbit", lambda: PBitOptimizer(lr=0.01, beta0=2.0, tau=150), flops_per_step=fp),
        OptimizerSpec("adam", lambda: Adam(lr=0.1), flops_per_step=fp),
    ]


def _experiment(max_iter=100, n_runs=3, seed=42):
    return Experiment(
        {
            "functions": [Rastrigin(dim=2), Rosenbrock(dim=2)],
            "noises": [None, QuantizeNoise(bits=4, stochastic=True)],
            "optimizers": _specs(),
            "max_iter": max_iter,
            "n_runs": n_runs,
            "seed": seed,
        }
    )


def test_run_produces_cells():
    ex = _experiment(max_iter=50, n_runs=2)
    rep = ex.run()
    # functions(2) x noises(2) x optimizers(2) = 8 cells
    assert len(rep.cells()) == 8
    assert rep.final_best_loss() > 0


def test_reproducible_same_seed():
    rep1 = _experiment().run()
    rep2 = _experiment().run()
    for c1, c2 in zip(rep1.cells(), rep2.cells()):
        assert c1.final_best_loss_mean == pytest.approx(c2.final_best_loss_mean)


def test_config_hash_stable():
    assert _experiment().hash == _experiment().hash
    ex_diff = Experiment({"seed": 43, "functions": [Rastrigin()], "noises": [None], "optimizers": _specs(), "max_iter": 100, "n_runs": 3})
    assert ex_diff.hash != _experiment().hash


def test_common_random_numbers_pair_optimizers_within_a_cell():
    """Every optimizer in a cell must start at the same point.

    The runner keys start position and gradient noise on the problem only, not
    on the optimizer name, so optimizers in a cell share a start and a
    perturbation. This is what lets per-run results be compared across
    optimizers. Keying them on the name too would break the pairing while
    leaving every result individually well-formed.
    """
    starts = {}

    class _Recorder:
        """Stands in for an optimizer and records the start it was handed."""

        def __init__(self, name):
            self.name = name

        def reset(self, rng=None):
            pass

        def step(self, x, grad, t):
            starts.setdefault(self.name, x.copy())
            return x - 0.01 * grad

    Experiment(
        {
            "functions": [Rastrigin(dim=2)],
            "noises": [None],
            "optimizers": [
                OptimizerSpec("alpha", lambda: _Recorder("alpha")),
                OptimizerSpec("beta", lambda: _Recorder("beta")),
                OptimizerSpec("gamma", lambda: _Recorder("gamma")),
            ],
            "max_iter": 1,
            "n_runs": 1,
            "seed": 7,
        }
    ).run()
    assert len(starts) == 3
    first = starts["alpha"]
    for name in ("beta", "gamma"):
        assert np.allclose(starts[name], first), f"{name} started elsewhere"


def test_start_position_shared_across_noise_conditions():
    """Clean and noisy runs of one optimizer must share a start.

    Cross-noise degradation comparisons pair on the start position; keying the
    init stream on the noise name would silently break that pairing.
    """
    starts = {}

    class _Recorder:
        def reset(self, rng=None):
            pass

        def step(self, x, grad, t):
            starts.setdefault(self._noise_name, x.copy())
            return x

        _noise_name = ""

    rec = _Recorder()

    def factory():
        return rec

    for noise in (None, QuantizeNoise(bits=2, stochastic=True)):
        rec._noise_name = "clean" if noise is None else noise.name
        Experiment(
            {
                "functions": [Rastrigin(dim=2)],
                "noises": [noise],
                "optimizers": [OptimizerSpec("x", factory)],
                "max_iter": 1,
                "n_runs": 1,
                "seed": 9,
            }
        ).run()

    assert np.allclose(starts["clean"], starts["QuantizeNoise_bits=2_stochastic=True"])


def test_private_rng_stream_identical_across_noise_conditions():
    """A stochastic optimizer must reuse the same coins across noise conditions.

    Cross-noise comparisons isolate the noise as the only changing input. If the
    opt stream were keyed on the noise name, the injected draws below would
    differ between the clean and noisy runs.
    """
    from pbit.optim import Langevin as _Lang

    class _TappingLangevin(_Lang):
        """Records the injected noise of every step."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self.tapped: list[np.ndarray] = []

        def step(self, x, grad, t):
            noise = np.sqrt(2 * self.lr * self.temperature) * self._rng.standard_normal(x.shape)
            self.tapped.append(noise.copy())
            return x - self.lr * grad + noise

    draws = {}
    for label, noise in (("clean", None), ("gauss", GaussianNoise(sigma=0.5))):
        opt = _TappingLangevin(lr=0.01, temperature=1.0)
        Experiment(
            {
                "functions": [Rosenbrock(dim=2)],
                "noises": [noise],
                "optimizers": [OptimizerSpec("x", lambda: opt)],
                "max_iter": 8,
                "n_runs": 1,
                "seed": 13,
            }
        ).run()
        draws[label] = opt.tapped

    assert len(draws["clean"]) == len(draws["gauss"]) == 8
    for a, b in zip(draws["clean"], draws["gauss"]):
        assert np.allclose(a, b)


def test_private_rng_stream_still_differs_per_optimizer():
    """Pairing must not collapse an optimizer's own randomness into another's.

    Langevin injects noise from its private stream, so two differently named
    Langevin instances in one cell must not draw identical noise.
    """
    rep = Experiment(
        {
            "functions": [Rosenbrock(dim=2)],
            "noises": [None],
            "optimizers": [
                OptimizerSpec("l1", lambda: Langevin(lr=0.01, temperature=1.0)),
                OptimizerSpec("l2", lambda: Langevin(lr=0.01, temperature=1.0)),
            ],
            "max_iter": 1,
            "n_runs": 1,
            "seed": 11,
        }
    ).run()
    assert not np.allclose(rep.rows[0].current_history, rep.rows[1].current_history)


def test_best_is_monotone_per_run():
    ex = _experiment(max_iter=40, n_runs=2)
    rep = ex.run()
    for r in rep._rows:  # noqa: SLF001
        assert np.all(np.diff(r.best_history) <= 1e-12)


def test_csv_export_ascii_only():
    rep = _experiment(max_iter=20, n_runs=2).run()
    path = rep.to_csv(Path("out") / "test.csv")
    assert path.exists()
    text = path.read_text(encoding="ascii")
    assert "Rastrigin" in text


def test_json_export_metadata():
    rep = _experiment(max_iter=20, n_runs=1).run()
    path = rep.to_json(Path("out") / "test.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "config_hash" in data
    assert "cells" in data
    assert data["metadata"]["pbit_version"]


def test_compare_reports_same():
    from pbit.bench import Experiment as E
    from pbit.bench import QuantizeNoise as QN
    from pbit.bench import Rastrigin as R
    from pbit.bench.specs import OptimizerSpec as OS
    from pbit.optim import Adam as Adamopt

    def make():
        return E(
            {
                "functions": [R(dim=2)],
                "noises": [None, QN(bits=2, stochastic=True)],
                "optimizers": [OS("adam", lambda: Adamopt(lr=0.05))],
                "max_iter": 10,
                "n_runs": 1,
                "seed": 1,
            }
        )

    a = make().run().to_json(Path("out") / "a.json")
    b = make().run().to_json(Path("out") / "b.json")
    diff = compare_reports(a, b)
    assert diff.config_match is True
    assert diff.software_match is True
    assert diff.max_loss_delta < 1e-9


def test_compare_reports_config_diff():
    d1 = _experiment(seed=1, max_iter=10, n_runs=1).run().to_json(Path("out") / "d1.json")
    d2 = _experiment(seed=2, max_iter=10, n_runs=1).run().to_json(Path("out") / "d2.json")
    diff = compare_reports(d1, d2)
    assert diff.config_match is False


def test_every_optimizer_family_gets_the_same_objective_budget():
    """One objective evaluation per iteration, regardless of optimizer family.

    Ask/tell optimizers used to evaluate each candidate twice (once in the
    runner, once inside tell()), doubling their budget and making every
    reported comparison against gradient optimizers invalid.
    """
    from pbit.optim import EvolutionStrategy, SimulatedAnnealing

    max_iter = 50
    counts = {}
    for name, factory in [
        ("pbit", lambda: PBitOptimizer(lr=0.05, tau=300)),
        ("adam", lambda: Adam(lr=0.05)),
        ("simanneal", lambda: SimulatedAnnealing(x0=np.zeros(2), seed=1)),
        ("evostrat", lambda: EvolutionStrategy(x0=np.zeros(2), lam=4, seed=1)),
    ]:
        rep = Experiment(
            {
                "functions": [Rastrigin(dim=2)],
                "noises": [None],
                "optimizers": [OptimizerSpec(name, factory)],
                "max_iter": max_iter,
                "n_runs": 2,
                "seed": 42,
            }
        ).run()
        counts[name] = rep.cells()[0].eval_count_mean

    assert set(counts.values()) == {float(max_iter)}, counts


def test_uncertainty_accompany_every_reported_mean():
    """sd and a confidence interval must travel with the mean, not replace it."""
    rep = _experiment(max_iter=30, n_runs=6).run()
    for c in rep.cells():
        assert np.isfinite(c.final_best_loss_sd)
        assert c.final_best_loss_ci_lo <= c.final_best_loss_mean <= c.final_best_loss_ci_hi
        assert c.final_best_loss_sd > 0
        assert np.isfinite(c.final_current_loss_sd)


def test_noisy_vs_clean_separated():
    ex = Experiment(
        {
            "functions": [Rosenbrock(dim=2)],
            "noises": [None, GaussianNoise(sigma=0.5)],
            "optimizers": _specs()[:1],
            "max_iter": 30,
            "n_runs": 1,
            "seed": 1,
        }
    )
    rep = ex.run()
    c = rep.get_cell("Rosenbrock", "clean", "pbit")
    assert c is not None
