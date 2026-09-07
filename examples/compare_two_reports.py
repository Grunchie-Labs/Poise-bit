"""compare_two_reports.py - diff two experiment reports.

Runs the same small experiment twice and compares the resulting JSON reports,
demonstrating the reproducibility/diff helper. Loss histories should agree
bit-for-bit; only wall-time differs across runs/machines.

Run:
    python examples/compare_two_reports.py
"""

from __future__ import annotations

from pathlib import Path

from pbit.bench import Experiment, Rastrigin
from pbit.bench.compare import compare_reports
from pbit.bench.specs import OptimizerSpec
from pbit.optim import Adam, PBitOptimizer


def flops(dim: int) -> int:
    return dim * 6


def run(out_path: Path, seed: int) -> None:
    ex = Experiment(
        {
            "functions": [Rastrigin(dim=2)],
            "noises": [None],
            "optimizers": [
                OptimizerSpec("pbit", lambda: PBitOptimizer(lr=0.05, tau=300, seed=1), flops_per_step=flops),
                OptimizerSpec("adam", lambda: Adam(lr=0.05), flops_per_step=flops),
            ],
            "max_iter": 300,
            "n_runs": 4,
            "seed": seed,
        }
    )
    report = ex.run()
    report.to_json(out_path)
    print(f"wrote {out_path}")


def main() -> None:
    out = Path("out")
    out.mkdir(exist_ok=True)
    a, b = out / "rep_a.json", out / "rep_b.json"

    # Identical seed -> same config, should be loss-reproducible.
    run(a, seed=42)
    run(b, seed=42)
    print("\n--- diff (same seed) ---")
    print(compare_reports(a, b).summary())

    # Different seed -> different config_hash (seed is part of the reproducible
    # config). Losses legitimately diverge because x-init & noise streams differ:
    # this example is about detecting that config change, not about wall-time.
    run(b, seed=7)
    print("\n--- diff (different seed) ---")
    print(compare_reports(a, b).summary())


if __name__ == "__main__":
    main()
