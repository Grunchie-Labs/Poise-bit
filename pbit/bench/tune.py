"""Equal-budget hyperparameter tuning.

The optimizer comparison is only meaningful if each method is set to its own
best configuration. Hand-picked learning rates favour whichever method the
author happened to tune for: Adam and SGD have very different scale
requirements, so a single shared rate under-serves one of them.

``tune_optimizer`` searches a declared grid for one (function, noise) cell and
returns the winner together with the score of every point in the grid. Keeping
the full grid matters: a reader who cannot see what else was tried cannot tell a
genuine optimum from a lucky pick.

Two rules keep the selection honest:

- **Tuning seeds are disjoint from evaluation seeds.** Selecting a configuration
  on the seeds you later report leaks the selection into the result. The seed
  offset is checked by :func:`assert_seeds_disjoint`.
- **The selection criterion is declared up front** and defaults to terminal
  current loss, not the min-over-trajectory, which improves under noise for
  reasons unrelated to optimization quality.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pbit.bench.experiment import Experiment
from pbit.bench.specs import FunctionSpec, NoiseSpec, OptimizerSpec

# Criteria a grid may be selected on. Terminal current loss is the default
# because it answers "where did the optimizer end up".
CRITERIA = {
    "final_current_loss_mean": lambda c: c.final_current_loss_mean,
    "final_best_loss_mean": lambda c: c.final_best_loss_mean,
    "auc_current_loss": lambda c: c.auc_current_loss,
    "auc_best_loss": lambda c: c.auc_best_loss,
    "success_rate": lambda c: -c.success_rate,
}


@dataclass
class GridPoint:
    """One candidate configuration and how it scored during tuning."""

    params: dict[str, Any]
    score: float
    metric: str
    n_tune_runs: int
    final_current_loss_mean: float
    final_best_loss_mean: float
    auc_current_loss: float
    success_rate: float


@dataclass
class TuningResult:
    """Selected configuration plus the complete grid that produced it."""

    optimizer: str
    function: str
    dim: int
    noise: str
    metric: str
    selected: dict[str, Any]
    selected_score: float
    grid: list[GridPoint] = field(default_factory=list)
    tune_seed: int = 0
    n_tune_runs: int = 0
    n_eval_runs: int = 0
    max_iter: int = 0

    @property
    def n_points(self) -> int:
        return len(self.grid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "function": self.function,
            "dim": self.dim,
            "noise": self.noise,
            "metric": self.metric,
            "selected": self.selected,
            "selected_score": self.selected_score,
            "n_grid_points": self.n_points,
            "tune_seed": self.tune_seed,
            "n_tune_runs": self.n_tune_runs,
            "n_eval_runs": self.n_eval_runs,
            "max_iter": self.max_iter,
            "grid": [
                {
                    "params": g.params,
                    "score": g.score,
                    "final_current_loss_mean": g.final_current_loss_mean,
                    "final_best_loss_mean": g.final_best_loss_mean,
                    "auc_current_loss": g.auc_current_loss,
                    "success_rate": g.success_rate,
                }
                for g in self.grid
            ],
        }


def assert_seeds_disjoint(tune_seed: int, eval_seed: int) -> None:
    """Refuse to tune and evaluate on the same master seed.

    Selecting a configuration and then reporting it on the same seeds hides how
    much of the result came from the search rather than the configuration.
    """
    if tune_seed == eval_seed:
        raise ValueError(
            f"tuning seed {tune_seed} equals evaluation seed {eval_seed}; "
            "the selection would leak into the reported result"
        )


def grid_from_values(name: str, values: Sequence[Any]) -> list[dict[str, Any]]:
    """Expand a single hyperparameter into a one-dimensional grid."""
    return [{name: v} for v in values]


def cross(*grids: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cartesian product of parameter dictionaries, in a stable order."""
    out: list[dict[str, Any]] = [{}]
    for g in grids:
        out = [dict(base, **item) for base in out for item in g]
    return out


def tune_optimizer(
    name: str,
    factory: Callable[[dict[str, Any]], Any],
    grid: Sequence[dict[str, Any]],
    function: FunctionSpec,
    noise: NoiseSpec | None,
    max_iter: int,
    n_tune_runs: int,
    tune_seed: int,
    metric: str = "final_current_loss_mean",
) -> TuningResult:
    """Search ``grid`` for the configuration minimizing ``metric``.

    ``factory`` maps a parameter dict to a fresh optimizer instance. The whole
    grid is scored on ``n_tune_runs`` runs from ``tune_seed``; the caller is
    responsible for using a different seed when it evaluates.
    """
    if metric not in CRITERIA:
        raise ValueError(f"unknown metric {metric!r}; options: {sorted(CRITERIA)}")
    if not grid:
        raise ValueError("grid must not be empty")
    score_of = CRITERIA[metric]

    points: list[GridPoint] = []
    for params in grid:
        spec = OptimizerSpec(name, lambda p=params: factory(p))
        rep = Experiment(
            {
                "functions": [function],
                "noises": [noise],
                "optimizers": [spec],
                "max_iter": max_iter,
                "n_runs": n_tune_runs,
                "seed": tune_seed,
            }
        ).run()
        cell = rep.cells()[0]
        points.append(
            GridPoint(
                params=dict(params),
                score=float(score_of(cell)),
                metric=metric,
                n_tune_runs=n_tune_runs,
                final_current_loss_mean=cell.final_current_loss_mean,
                final_best_loss_mean=cell.final_best_loss_mean,
                auc_current_loss=cell.auc_current_loss,
                success_rate=cell.success_rate,
            )
        )

    # NaN scores (a diverged configuration) must sort last, not win by accident.
    finite = [p for p in points if np.isfinite(p.score)]
    best = min(finite, key=lambda p: p.score) if finite else min(
        points, key=lambda p: (np.isfinite(p.score), p.score)
    )

    return TuningResult(
        optimizer=name,
        function=function.name,
        dim=function.dim,
        noise=noise.name if noise is not None else "clean",
        metric=metric,
        selected=dict(best.params),
        selected_score=float(best.score),
        grid=points,
        tune_seed=tune_seed,
        n_tune_runs=n_tune_runs,
        max_iter=max_iter,
    )
