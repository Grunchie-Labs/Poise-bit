"""run_benchmark.py — full reproducible optimizer benchmark.

Compares PBit against AI-relevant baselines across multiple functions and
noise modes, exports CSV/JSON, and prints a summary table.

Run:
    python examples/run_benchmark.py
"""

from __future__ import annotations

from pathlib import Path

from pbit.bench import (
    Ackley,
    CorruptionNoise,
    Experiment,
    GaussianNoise,
    QuantizeNoise,
    Rastrigin,
    Rosenbrock,
)
from pbit.bench.specs import OptimizerSpec
from pbit.optim import (
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


def flops(dim: int) -> int:
    # Declared PROXY cost (kind="proxy"), never combined with measured wall-time.
    return dim * 6


def optimizers() -> list[OptimizerSpec]:
    defs = [
        ("pbit", lambda: PBitOptimizer(lr=0.05, tau=300, seed=1)),
        ("pbit_floor", lambda: PBitOptimizer(lr=0.05, tau=300, step_size="floor", floor=1e-3, seed=1)),
        ("adam", lambda: Adam(lr=0.05)),
        ("adamw", lambda: AdamW(lr=0.05)),
        ("signsgd", lambda: SignSGD(lr=0.05)),
        ("lion", lambda: Lion(lr=0.01)),
        ("sgd", lambda: SGD(lr=0.05)),
        ("momentum", lambda: Momentum(lr=0.05)),
        ("rmsprop", lambda: RMSProp(lr=0.05)),
        ("langevin", lambda: Langevin(lr=0.05)),
    ]
    return [OptimizerSpec(name, factory, flops_per_step=flops, params={}) for name, factory in defs]


def main() -> None:
    out = Path("out")
    ex = Experiment(
        {
            "functions": [Rastrigin(dim=2), Ackley(dim=2), Rosenbrock(dim=2)],
            "noises": [
                None,
                GaussianNoise(sigma=0.5),
                CorruptionNoise(p=0.2),
                QuantizeNoise(bits=4, stochastic=False),
                QuantizeNoise(bits=4, stochastic=True),
            ],
            "optimizers": optimizers(),
            "max_iter": 500,
            "n_runs": 5,
            "seed": 42,
        }
    )
    print(f"Running {len(ex.functions)} functions x {len(ex.noises)} noises x "
          f"{len(ex.optimizers)} optimizers x {ex.n_runs} runs ...")
    report = ex.run()
    print("\n" + report.summary())

    report.to_csv(out / "run_benchmark.csv")
    report.to_json(out / "run_benchmark.json")
    print(f"\nSaved artifacts to {out.resolve()}")
    print(f"config_hash = {ex.hash}")


if __name__ == "__main__":
    main()
