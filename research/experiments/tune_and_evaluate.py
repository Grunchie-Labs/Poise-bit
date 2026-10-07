"""Tune every optimizer per cell, then evaluate on held-out seeds.

This is the measurement core of the research layer. Every comparison reported
in CLAIMS.md uses the configurations selected here, so no optimizer is judged
at a hand-picked learning rate.

Protocol:

1. For each (function, dimension, noise) cell, search each optimizer's grid on
   ``n_tune_runs`` runs from ``tune_seed``. The selection criterion is terminal
   current loss. The full grid and every score are recorded, so the selection
   can be audited rather than trusted.
2. Evaluate the selected configuration of every optimizer on ``n_eval_runs``
   runs from ``eval_seed``, which is required to differ from ``tune_seed``.
   Per-run terminal values are recorded so claims.py can run paired statistics
   without re-running any optimization.

Output: ``results/tuned.json`` (tuning records + eval series + metadata).

Usage:
    python research/experiments/tune_and_evaluate.py
    python research/experiments/tune_and_evaluate.py --eval-runs 50 --iters 300
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pbit.bench import (  # noqa: E402
    Ackley,
    CorruptionNoise,
    Experiment,
    GaussianNoise,
    QuantizeNoise,
    Rastrigin,
    Rosenbrock,
    SignNoise,
    assert_seeds_disjoint,
    cross,
    grid_from_values,
    iter_to_threshold,
    tune_optimizer,
)
from pbit.bench.specs import OptimizerSpec  # noqa: E402
from pbit.optim import (  # noqa: E402
    SGD,
    Adam,
    AdamW,
    Langevin,
    Lion,
    Momentum,
    PBitOptimizer,
    RMSProp,
    SignSGD,
)

TUNE_SEED = 1000
EVAL_SEED = 0
THRESHOLD = 1.0

FACTORIES = {
    "pbit": lambda p: PBitOptimizer(lr=p["lr"], tau=p["tau"]),
    "pbit_floor": lambda p: PBitOptimizer(
        lr=p["lr"], tau=p["tau"], step_size="floor", floor=1e-3
    ),
    "sgd": lambda p: SGD(lr=p["lr"]),
    "momentum": lambda p: Momentum(lr=p["lr"]),
    "adam": lambda p: Adam(lr=p["lr"]),
    "adamw": lambda p: AdamW(lr=p["lr"]),
    "rmsprop": lambda p: RMSProp(lr=p["lr"]),
    "signsgd": lambda p: SignSGD(lr=p["lr"]),
    "lion": lambda p: Lion(lr=p["lr"]),
    "langevin": lambda p: Langevin(lr=p["lr"], temperature=0.1),
}

GRIDS = {
    "pbit": cross(grid_from_values("lr", [0.005, 0.01, 0.05, 0.1]),
                  grid_from_values("tau", [100, 300, 1000])),
    "pbit_floor": cross(grid_from_values("lr", [0.005, 0.01, 0.05, 0.1]),
                        grid_from_values("tau", [100, 300, 1000])),
    "sgd": grid_from_values("lr", [0.001, 0.01, 0.05, 0.1, 0.5]),
    "momentum": grid_from_values("lr", [0.001, 0.01, 0.05, 0.1, 0.5]),
    "adam": grid_from_values("lr", [0.001, 0.01, 0.05, 0.1, 0.5]),
    # AdamW gets its defining knob. With weight_decay=0 it is identical to
    # Adam, which would put a duplicate optimizer in the study.
    "adamw": cross(grid_from_values("lr", [0.001, 0.01, 0.05, 0.1, 0.5]),
                   grid_from_values("weight_decay", [0.0, 0.01, 0.1])),
    "rmsprop": grid_from_values("lr", [0.001, 0.01, 0.05, 0.1, 0.5]),
    "signsgd": grid_from_values("lr", [0.001, 0.005, 0.01, 0.05, 0.1]),
    "lion": grid_from_values("lr", [0.001, 0.003, 0.01, 0.03, 0.1]),
    "langevin": grid_from_values("lr", [0.001, 0.01, 0.05, 0.1, 0.5]),
}


def make_cells() -> list[tuple[str, int, object]]:
    """The cell matrix: all noise conditions at dim=2, clean only at dim=10."""
    noises_d2 = [
        None,
        GaussianNoise(sigma=0.5),
        CorruptionNoise(p=0.2),
        QuantizeNoise(bits=4, stochastic=True),
        SignNoise(),
    ]
    cells = []
    for fn in (Rastrigin, Ackley, Rosenbrock):
        for noise in noises_d2:
            cells.append((fn.__name__, 2, noise))
        cells.append((fn.__name__, 10, None))
    return cells


def make_function(name: str, dim: int):
    return {"Rastrigin": Rastrigin, "Ackley": Ackley, "Rosenbrock": Rosenbrock}[name](dim=dim)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tune-runs", type=int, default=5)
    ap.add_argument("--eval-runs", type=int, default=50)
    ap.add_argument("--tune-seed", type=int, default=TUNE_SEED)
    ap.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    ap.add_argument("--out", type=str, default="results/tuned.json")
    args = ap.parse_args()

    assert_seeds_disjoint(args.tune_seed, args.eval_seed)
    cells = make_cells()
    print(f"cells: {len(cells)}, optimizers: {len(FACTORIES)}, "
          f"grid points per optimizer: {[len(g) for g in GRIDS.values()]}")

    t_start = time.perf_counter()
    # Tune on clean cells only, then transfer the selected config to every
    # noise condition of the same function and dimension. Tuning per noise
    # cell lets each optimizer adapt to the noise, and where the tuning
    # landscape is flat the per-cell selections are effectively a lottery:
    # a "noise effect" computed from two different configs measures the
    # lottery, not the noise. With transfer, clean and noisy runs share one
    # config and the shift isolates the noise.
    tuning: dict[str, dict] = {}
    for fn_name, dim, noise in cells:
        if noise is not None:
            continue
        fn = make_function(fn_name, dim)
        cell_key = f"{fn_name}/dim={dim}/clean"
        tuning[cell_key] = {}
        print(f"\n[tune] {cell_key}")
        for opt_name, factory in FACTORIES.items():
            res = tune_optimizer(
                opt_name,
                factory,
                GRIDS[opt_name],
                fn,
                None,
                max_iter=args.iters,
                n_tune_runs=args.tune_runs,
                tune_seed=args.tune_seed,
            )
            tuning[cell_key][opt_name] = res.to_dict()
            print(f"  {opt_name:<12} selected={res.selected} "
                  f"score={res.selected_score:.4f}")

    t_tune = time.perf_counter()
    print(f"\ntuning took {t_tune - t_start:.1f}s; evaluating on held-out seeds ...")

    evaluation: dict[str, dict] = {}
    eval_configs: dict[str, dict] = {}
    for fn_name, dim, noise in cells:
        fn = make_function(fn_name, dim)
        cell_key = f"{fn_name}/dim={dim}/{noise.name if noise else 'clean'}"
        clean_key = f"{fn_name}/dim={dim}/clean"
        eval_configs[cell_key] = {
            opt: tuning[clean_key][opt]["selected"] for opt in FACTORIES
        }
        specs = [
            OptimizerSpec(
                opt_name,
                lambda p=tuning[clean_key][opt_name]["selected"], f=factory: f(p),
            )
            for opt_name, factory in FACTORIES.items()
        ]
        rep = Experiment(
            {
                "functions": [fn],
                "noises": [noise],
                "optimizers": specs,
                "max_iter": args.iters,
                "n_runs": args.eval_runs,
                "seed": args.eval_seed,
                "threshold": THRESHOLD,
            }
        ).run()

        per_opt: dict[str, dict] = {}
        for row in rep.rows:
            bucket = per_opt.setdefault(
                row.optimizer,
                {"final_current": {}, "final_best": {}, "hit_time": {}},
            )
            bucket["final_current"][row.run_id] = float(row.current_history[-1])
            bucket["final_best"][row.run_id] = float(row.best_history[-1])
            bucket["hit_time"][row.run_id] = iter_to_threshold(row.best_history, THRESHOLD)
        evaluation[cell_key] = per_opt
        print(f"[eval] {cell_key} done")

    payload = {
        "metadata": {
            "iters": args.iters,
            "tune_runs": args.tune_runs,
            "eval_runs": args.eval_runs,
            "tune_seed": args.tune_seed,
            "eval_seed": args.eval_seed,
            "threshold": THRESHOLD,
            "criterion": "final_current_loss_mean",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "wall_seconds": time.perf_counter() - t_start,
        },
        "grids": {k: v for k, v in GRIDS.items()},
        "tuning": tuning,
        "eval_configs": eval_configs,
        "evaluation": evaluation,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
