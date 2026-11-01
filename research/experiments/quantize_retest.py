"""Quantization retest on Sphere: the claim C1 could not test, measured where it can be.

Claim C1 was withdrawn as untestable: on Rastrigin the runner's gradient
clip at 5.0 fires on every step, and clip(quantized) == clip(raw), so there
is nothing for quantization to change. This script re-runs the comparison on
Sphere, a convex quadratic whose gradients (|g| <= 4 on the init box) never
reach the clip, so the comparison measures quantization.

Protocol mirrors the main study: tune on clean Sphere, transfer the selected
config to quantized cells, evaluate on held-out seeds, compare in pairs.
Three bit-widths (1, 4, 8-bit stochastic) against clean, for PBit plus Adam,
SGD, and SignSGD controls. Writes results/quantize_retest.json.

Usage:
    python research/experiments/quantize_retest.py
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
    Experiment,
    QuantizeNoise,
    Sphere,
    assert_seeds_disjoint,
    cross,
    grid_from_values,
    tune_optimizer,
)
from pbit.bench.specs import OptimizerSpec  # noqa: E402
from pbit.bench.stats import describe, holm_bonferroni, paired_compare  # noqa: E402
from pbit.optim import SGD, Adam, PBitOptimizer, SignSGD  # noqa: E402

TUNE_SEED = 1000
EVAL_SEED = 0
ITERS = 100
DIM = 10

FACTORIES = {
    "pbit": lambda p: PBitOptimizer(lr=p["lr"], tau=p["tau"]),
    "adam": lambda p: Adam(lr=p["lr"]),
    "sgd": lambda p: SGD(lr=p["lr"]),
    "signsgd": lambda p: SignSGD(lr=p["lr"]),
}

GRIDS = {
    "pbit": cross(
        grid_from_values("lr", [0.005, 0.01, 0.05, 0.1]),
        grid_from_values("tau", [100, 300, 1000]),
    ),
    "adam": grid_from_values("lr", [0.001, 0.01, 0.05, 0.1, 0.5]),
    "sgd": grid_from_values("lr", [0.001, 0.01, 0.05, 0.1, 0.5]),
    "signsgd": grid_from_values("lr", [0.001, 0.005, 0.01, 0.05, 0.1]),
}

BITS = [1, 4, 8]


def run_premise_check(iters: int, eval_seed: int) -> list[int]:
    """Verify the clip (nearly) never fires on Sphere clean runs."""
    fn = Sphere(dim=DIM)
    probe = Experiment(
        {
            "functions": [fn],
            "noises": [None, QuantizeNoise(bits=4, stochastic=True)],
            "optimizers": [OptimizerSpec("pbit", lambda: PBitOptimizer(lr=0.05, tau=300))],
            "max_iter": iters,
            "n_runs": 10,
            "seed": eval_seed,
        }
    ).run()
    counts = [int(r.clip_count) for r in probe.rows]
    print(f"[premise] clip events per run on Sphere dim={DIM}: {counts}")
    return counts


def run_tuning(fn, iters: int, tune_runs: int, tune_seed: int) -> dict:
    """Tune every optimizer on the clean Sphere cell; return records."""
    tuning: dict[str, dict] = {}
    print(f"\n[tune] Sphere/dim={DIM}/clean")
    for opt_name, factory in FACTORIES.items():
        res = tune_optimizer(
            opt_name, factory, GRIDS[opt_name], fn, None,
            max_iter=iters, n_tune_runs=tune_runs, tune_seed=tune_seed,
        )
        tuning[opt_name] = res.to_dict()
        print(f"  {opt_name:<10} selected={res.selected} score={res.selected_score:.6f}")
    return tuning


def run_evaluation(fn, tuning: dict, iters: int, eval_runs: int, eval_seed: int) -> dict:
    """Evaluate clean + each bit-width; return ordered per-run series."""
    noises = [None] + [QuantizeNoise(bits=b, stochastic=True) for b in BITS]
    series: dict[str, dict] = {}
    for noise in noises:
        key = "clean" if noise is None else noise.name
        specs = [
            OptimizerSpec(opt, lambda p=tuning[opt]["selected"], f=f: f(p))
            for opt, f in FACTORIES.items()
        ]
        rep = Experiment(
            {
                "functions": [fn],
                "noises": [noise],
                "optimizers": specs,
                "max_iter": iters,
                "n_runs": eval_runs,
                "seed": eval_seed,
            }
        ).run()
        bucket: dict[str, dict] = {}
        bucket_auc: dict[str, dict] = {}
        for row in rep.rows:
            bucket.setdefault(row.optimizer, {})[row.run_id] = float(row.current_history[-1])
            bucket_auc.setdefault(row.optimizer, {})[row.run_id] = float(
                np.mean(np.abs(row.current_history))
            )
        series[key] = {
            opt: np.array([v[k] for k in sorted(v, key=int)]) for opt, v in bucket.items()
        }
        series[key].update({
            opt + ":auc": np.array([v[k] for k in sorted(v, key=int)])
            for opt, v in bucket_auc.items()
        })
        print(f"[eval] Sphere/dim={DIM}/{key} done")
    return series


def build_claims(series: dict) -> list[dict]:
    """Per (optimizer, bits): paired noisy-minus-clean shift, Holm family.

    Two readers per comparison: the pre-registered terminal current loss
    (matches every other claim in the study) and the trajectory AUC (mean
    current loss over the run). On Sphere the terminal reader can saturate
    at machine precision for fast solvers, so the AUC reader carries the
    discriminative signal; both are reported, and a comparison counts as
    resolved only if the readers agree in direction.
    """
    claims: list[dict] = []
    comparators = []
    targets = []
    for opt in FACTORIES:
        for b in BITS:
            noisy_t = series[f"QuantizeNoise_bits={b}_stochastic=True"][opt]
            clean_t = series["clean"][opt]
            noisy_a = series[f"QuantizeNoise_bits={b}_stochastic=True"][opt + ":auc"]
            clean_a = series["clean"][opt + ":auc"]
            cmp_t = paired_compare(noisy_t, clean_t, f"{opt} {b}-bit vs clean (terminal)")
            cmp_a = paired_compare(noisy_a, clean_a, f"{opt} {b}-bit vs clean (AUC)")
            entry = {
                "optimizer": opt,
                "bits": b,
                "comparison": cmp_t.to_dict(),
                "clean": describe(clean_t, f"{opt} clean"),
                "noisy": describe(noisy_t, f"{opt} {b}-bit"),
                "comparison_auc": cmp_a.to_dict(),
                "clean_auc": describe(clean_a, f"{opt} clean AUC"),
                "noisy_auc": describe(noisy_a, f"{opt} {b}-bit AUC"),
            }
            claims.append(entry)
            comparators.append(cmp_t)
            targets.append(entry["comparison"])
    for holder, adj in zip(targets, holm_bonferroni(comparators)):
        holder["p_value_holm"] = adj.p_value
    # Second family: the AUC reader gets its own Holm correction (12
    # trajectory contrasts). Recorded alongside the terminal family's values.
    from pbit.bench.stats import Comparison as _Comparison

    _keys = ("label", "n", "mean_a", "mean_b", "mean_diff",
             "ci_lo", "ci_hi", "p_value", "sd_diff")
    _auc_comps = [_Comparison(**{k: e["comparison_auc"][k] for k in _keys}) for e in claims]
    _auc_targets = [e["comparison_auc"] for e in claims]
    for holder, adj in zip(_auc_targets, holm_bonferroni(_auc_comps)):
        holder["p_value_holm"] = adj.p_value
    for e in claims:
        c = e["comparison"]
        ca = e["comparison_auc"]
        print(f"  {e['optimizer']:<10} {e['bits']}-bit: terminal diff={c['mean_diff']:+.6g} "
              f"p={c['p_value']:.4f} holm={c['p_value_holm']:.4f} | "
              f"AUC diff={ca['mean_diff']:+.6g} CI=[{ca['ci_lo']:+.6g},{ca['ci_hi']:+.6g}] "
              f"p={ca['p_value']:.4f}")
    return claims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--tune-runs", type=int, default=10)
    ap.add_argument("--eval-runs", type=int, default=50)
    ap.add_argument("--tune-seed", type=int, default=TUNE_SEED)
    ap.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    ap.add_argument("--out", type=str, default="results/quantize_retest.json")
    args = ap.parse_args()

    assert_seeds_disjoint(args.tune_seed, args.eval_seed)
    fn = Sphere(dim=DIM)
    t_start = time.perf_counter()

    clip_counts = run_premise_check(args.iters, args.eval_seed)
    tuning = run_tuning(fn, args.iters, args.tune_runs, args.tune_seed)
    series = run_evaluation(fn, tuning, args.iters, args.eval_runs, args.eval_seed)
    claims = build_claims(series)

    payload = {
        "metadata": {
            "function": "Sphere", "dim": DIM, "iters": args.iters,
            "tune_runs": args.tune_runs, "eval_runs": args.eval_runs,
            "tune_seed": args.tune_seed, "eval_seed": args.eval_seed,
            "criterion": "final_current_loss_mean",
            "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(),
            "wall_seconds": time.perf_counter() - t_start,
        },
        "clip_premise_counts": clip_counts,
        "tuning": tuning,
        "claims": claims,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()


