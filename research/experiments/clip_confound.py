"""Measure the clipping confound in the quantization cell.

Claim C1 found the quantize-vs-clean comparison bit-identical on Rastrigin.
This script records why: the gradients in that cell exceed the clip range so
often that after clipping there is nothing left for quantization to change.
Quantization maps components to levels between the raw minimum and maximum,
preserving the endpoints, so a component that saturates the clip is identical
whether or not it was quantized first.

Writes results/clip_confound.json.

Usage:
    python research/experiments/clip_confound.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pbit.bench import Experiment, QuantizeNoise, Rastrigin  # noqa: E402
from pbit.bench.experiment import DEFAULT_CLIP  # noqa: E402
from pbit.bench.specs import OptimizerSpec  # noqa: E402
from pbit.optim import PBitOptimizer  # noqa: E402

N_RANDOM_GRADIENTS = 1000


def main() -> None:
    # 1. How often does the clip fire inside a real run of the C1 cell?
    rep = Experiment(
        {
            "functions": [Rastrigin(dim=2)],
            "noises": [None, QuantizeNoise(bits=4, stochastic=True)],
            "optimizers": [OptimizerSpec("pbit", lambda: PBitOptimizer(lr=0.05, tau=100))],
            "max_iter": 300,
            "n_runs": 10,
            "seed": 0,
        }
    ).run()
    clip_events = [int(r.clip_count) for r in rep.rows]
    iters = 300

    # 2. How large do Rastrigin gradients get on a domain-wide grid?
    xs = np.linspace(-5.12, 5.12, 400)
    gx, gy = np.meshgrid(xs, xs)
    grads = 2 * np.stack([gx, gy]) + 2 * np.pi * 10.0 * np.sin(2 * np.pi * np.stack([gx, gy]))
    max_component = float(np.abs(grads).max())

    # 3. Does clipping erase the quantization? Sample random gradients of
    # Rastrigin scale, quantize, clip both, and compare.
    rng = np.random.default_rng(0)
    identical = 0
    for _ in range(N_RANDOM_GRADIENTS):
        g = rng.uniform(-max_component, max_component, 2)
        q = QuantizeNoise(bits=4, stochastic=True).apply(g, rng)
        if np.allclose(np.clip(g, -5, 5), np.clip(q, -5, 5)):
            identical += 1

    payload = {
        "metadata": {
            "clip": DEFAULT_CLIP,
            "quantization_levels": 2**4,
            "n_runs_sampled": len(rep.rows),
            "random_gradients_tested": N_RANDOM_GRADIENTS,
        },
        "clip_events_per_run": clip_events,
        "iters_per_run": iters,
        "all_runs_fully_clipped": all(c == iters for c in clip_events),
        "max_gradient_component": max_component,
        "clip_erases_quantization_count": identical,
        "clip_erases_quantization_rate": identical / N_RANDOM_GRADIENTS,
    }
    out = Path("results/clip_confound.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
