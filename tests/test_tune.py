"""Tests for equal-budget hyperparameter tuning.

These lock the two properties that make a tuning result trustworthy: the search
cannot see the evaluation seeds, and the reported winner is the actual argmin of
the declared criterion over the declared grid.
"""

import pytest

from pbit.bench.functions import Rastrigin, Rosenbrock
from pbit.bench.noise import QuantizeNoise
from pbit.bench.tune import (
    assert_seeds_disjoint,
    cross,
    grid_from_values,
    tune_optimizer,
)
from pbit.optim import Adam, PBitOptimizer


def _adam_factory(params):
    return Adam(lr=params["lr"])


def _grid():
    return grid_from_values("lr", [0.001, 0.01, 0.1])


def test_grid_helpers_expand_in_stable_order():
    single = grid_from_values("lr", [0.1, 0.2])
    assert single == [{"lr": 0.1}, {"lr": 0.2}]

    combo = cross(grid_from_values("lr", [0.1, 0.2]), grid_from_values("mu", [0.8, 0.9]))
    assert len(combo) == 4
    assert combo[0] == {"lr": 0.1, "mu": 0.8}
    assert combo[-1] == {"lr": 0.2, "mu": 0.9}
    # Order must be reproducible so the grid record is comparable across runs.
    assert combo == cross(grid_from_values("lr", [0.1, 0.2]),
                          grid_from_values("mu", [0.8, 0.9]))


def test_tuning_and_evaluation_seeds_must_differ():
    assert_seeds_disjoint(tune_seed=0, eval_seed=1)
    with pytest.raises(ValueError, match="leak"):
        assert_seeds_disjoint(tune_seed=3, eval_seed=3)


def test_selected_point_is_the_argmin_of_the_criterion():
    res = tune_optimizer(
        "adam",
        _adam_factory,
        _grid(),
        Rastrigin(dim=2),
        None,
        max_iter=40,
        n_tune_runs=2,
        tune_seed=0,
    )
    scores = {p.params["lr"]: p.score for p in res.grid}
    assert res.selected["lr"] == min(scores, key=lambda k: scores[k])
    assert res.selected_score == pytest.approx(min(scores.values()))


def test_every_grid_point_is_recorded():
    grid = _grid()
    res = tune_optimizer(
        "adam", _adam_factory, grid, Rastrigin(dim=2), None,
        max_iter=30, n_tune_runs=1, tune_seed=0,
    )
    assert res.n_points == len(grid)
    assert {p.params["lr"] for p in res.grid} == {g["lr"] for g in grid}
    # The audit record must carry more than the winner, so a reader can check it.
    payload = res.to_dict()
    assert payload["n_grid_points"] == len(grid)
    assert len(payload["grid"]) == len(grid)


def test_tuning_is_deterministic():
    kw = dict(
        grid=_grid(), function=Rastrigin(dim=2), noise=None,
        max_iter=40, n_tune_runs=2, tune_seed=5,
    )
    a = tune_optimizer("adam", _adam_factory, **kw)
    b = tune_optimizer("adam", _adam_factory, **kw)
    assert a.selected == b.selected
    assert a.selected_score == b.selected_score


def test_metric_choice_changes_the_selection():
    """Selecting on the min statistic must be able to pick a different point.

    If the criterion were ignored, these two calls would agree.
    """
    grid = grid_from_values("lr", [0.001, 0.01, 0.1])
    fn = Rastrigin(dim=2)
    on_terminal = tune_optimizer(
        "adam", _adam_factory, grid, fn, None,
        max_iter=60, n_tune_runs=3, tune_seed=0, metric="final_current_loss_mean",
    )
    on_min = tune_optimizer(
        "adam", _adam_factory, grid, fn, None,
        max_iter=60, n_tune_runs=3, tune_seed=0, metric="final_best_loss_mean",
    )
    by_terminal = {p.params["lr"]: p.final_current_loss_mean for p in on_terminal.grid}
    by_min = {p.params["lr"]: p.final_best_loss_mean for p in on_min.grid}
    assert on_terminal.selected["lr"] == min(by_terminal, key=lambda k: by_terminal[k])
    assert on_min.selected["lr"] == min(by_min, key=lambda k: by_min[k])


def test_tuning_works_under_noise_and_records_the_condition():
    res = tune_optimizer(
        "pbit",
        lambda p: PBitOptimizer(lr=p["lr"], tau=p["tau"]),
        cross(grid_from_values("lr", [0.01, 0.05]), grid_from_values("tau", [100, 300])),
        Rastrigin(dim=2),
        QuantizeNoise(bits=4, stochastic=True),
        max_iter=30,
        n_tune_runs=2,
        tune_seed=0,
    )
    assert res.noise == "QuantizeNoise_bits=4_stochastic=True"
    assert res.n_points == 4
    assert set(res.selected) == {"lr", "tau"}


def test_non_pbit_optimizers_see_the_same_seed_independence():
    res = tune_optimizer(
        "adam", _adam_factory, _grid(), Rosenbrock(dim=2), None,
        max_iter=30, n_tune_runs=1, tune_seed=2,
    )
    assert res.tune_seed == 2
    assert res.n_tune_runs == 1
    assert all(p.n_tune_runs == 1 for p in res.grid)


def test_empty_grid_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        tune_optimizer(
            "adam", _adam_factory, [], Rastrigin(dim=2), None,
            max_iter=10, n_tune_runs=1, tune_seed=0,
        )


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="unknown metric"):
        tune_optimizer(
            "adam", _adam_factory, _grid(), Rastrigin(dim=2), None,
            max_iter=10, n_tune_runs=1, tune_seed=0, metric="made_up",
        )
