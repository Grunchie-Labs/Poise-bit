"""noise_sweep.py - low-precision gradient stress test.

The most AI/hardware-relevant example: sweep gradient quantization bit-width
(stochastic rounding) from 1 to 32 bits and compare how optimizers degrade.
Headline metric: robustness_ratio = noisy_final_best / clean_final_best.

Run:
    python examples/noise_sweep.py
"""

from __future__ import annotations

from pathlib import Path

from pbit.bench import Experiment, QuantizeNoise, Rastrigin
from pbit.bench.specs import OptimizerSpec
from pbit.optim import AdamW, Lion, PBitOptimizer, SignSGD


def flops(dim: int) -> int:
    return dim * 6


def main() -> None:
    out = Path("out")
    bits = [1, 2, 3, 4, 8, 16, 32]
    opt_defs = [
        ("pbit_prop", lambda: PBitOptimizer(lr=0.05, tau=300, seed=1)),
        ("pbit_floor", lambda: PBitOptimizer(lr=0.05, tau=300, step_size="floor", floor=1e-3, seed=1)),
        ("adamw", lambda: AdamW(lr=0.05)),
        ("signsgd", lambda: SignSGD(lr=0.05)),
        ("lion", lambda: Lion(lr=0.01)),
    ]
    optimizers = [OptimizerSpec(name, fac, flops_per_step=flops) for name, fac in opt_defs]

    # To compute robustness_ratio we need the clean baseline in the same report.
    noises = [None] + [QuantizeNoise(bits=b, stochastic=True) for b in bits]
    fn = Rastrigin(dim=2)
    ex = Experiment(
        {
            "functions": [fn],
            "noises": noises,
            "optimizers": optimizers,
            "max_iter": 500,
            "n_runs": 10,
            "seed": 42,
        }
    )
    print(f"Quantization sweep on {fn.name} (dim={fn.dim}) ...")
    report = ex.run()

    print("\nRobustness ratio (noisy_final_best / clean_final_best), lower=better:")
    header = f"{'optimizer':<12}" + "".join(f"{b:>8}bit" for b in bits)
    print(header)
    print("-" * len(header))
    for name, _ in opt_defs:
        clean = report.get_cell(fn.name, "clean", name).final_best_loss_mean
        row = f"{name:<12}"
        for b in bits:
            noisy = report.get_cell(fn.name, _label(b), name)
            ratio = noisy.final_best_loss_mean / max(clean, 1e-12)
            row += f"{ratio:>8.2f}"
        print(row)

    report.to_csv(out / "noise_sweep.csv")
    report.to_json(out / "noise_sweep.json")
    print(f"\nSaved artifacts to {out.resolve()}")


def _label(bits: int) -> str:
    return f"QuantizeNoise_bits={bits}_stochastic=True"


if __name__ == "__main__":
    main()
