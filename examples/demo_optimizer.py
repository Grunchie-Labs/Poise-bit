"""demo_optimizer.py — classic optimization with PBitOptimizer.

Shows PBit on Rastrigin and Rosenbrock with best-so-far tracking and the three
step-size modes.

Run:
    python examples/demo_optimizer.py
"""

from __future__ import annotations

import numpy as np

from pbit.bench.functions import Rastrigin, Rosenbrock
from pbit.optim import PBitOptimizer


def optimize(spec, opt, n_iter: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(*spec.init_bounds, size=spec.dim)
    best = float("inf")
    for t in range(n_iter):
        g = np.clip(spec.grad(x), *(-5.0, 5.0))
        x = opt.step(x, g, t)
        best = min(best, float(spec.evaluate(x)))
    return best


def main() -> None:
    print("=" * 60)
    print("PBit demo: probabilistic binary-direction optimizer")
    print("=" * 60)

    problems = [Rastrigin(dim=2), Rosenbrock(dim=2)]
    modes = [
        ("proportional (v2)", dict(step_size="proportional")),
        ("constant (v1)", dict(step_size="constant")),
        ("floor (escape)", dict(step_size="floor", floor=1e-3)),
    ]

    for spec in problems:
        print(f"\n--- {spec.name} (dim={spec.dim}, min at f={spec.minimum}) ---")
        for label, kw in modes:
            opt = PBitOptimizer(lr=0.05, tau=300, seed=1, **kw)
            best = optimize(spec, opt)
            print(f"  {label:<22} best-so-far = {best:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
