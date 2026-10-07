"""Power analysis over the recorded evaluation series.

Reads results/tuned.json and answers two questions with subsampling, without
re-running any optimization:

1. Power: for the confirmed sign-noise effect (claim C2), how often would a
   smaller study have detected it? A claim that needs 50 runs to reach
   significance is fragile at the n_runs=5 scale the earlier report used.
2. Size: for the zero-effect quantization comparison (claim C1), how often
   does a subsample call it an improvement or reach significance? A correct
   null should be called an improvement about half the time and reach
   significance about 5 percent of the time.

Usage:
    python research/experiments/power.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pbit.bench.stats import paired_compare  # noqa: E402

SIGN_CLEAN = "Rastrigin/dim=2/clean"
SIGN_NOISY = "Rastrigin/dim=2/SignNoise"
QUANT_NOISY = "Rastrigin/dim=2/QuantizeNoise_bits=4_stochastic=True"


def series(tuned: dict, cell: str, opt: str) -> np.ndarray:
    per_run = tuned["evaluation"][cell][opt]["final_current"]
    return np.array([per_run[k] for k in sorted(per_run, key=int)], dtype=np.float64)


def subsample(diffs: np.ndarray, sizes, draws: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        if n > diffs.size:
            continue
        neg, pos, sig = 0, 0, 0
        for _ in range(draws):
            idx = rng.choice(diffs.size, size=n, replace=False)
            cmp = paired_compare(diffs[idx], np.zeros(n), f"n={n}")
            if cmp.mean_diff < 0:
                neg += 1
            else:
                pos += 1
            if cmp.p_value < 0.05:
                sig += 1
        out.append({
            "n": n,
            "p_negative": neg / draws,
            "p_positive": pos / draws,
            "p_significant_at_0.05": sig / draws,
            "draws": draws,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", type=str, default="results/tuned.json")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--out", type=str, default="results/power.json")
    args = ap.parse_args()

    tuned = json.loads(Path(args.tuned).read_text(encoding="utf-8"))
    sizes = [5, 10, 20, 30, 50]

    # Power curve for the real effect: sign-noise shift difference, PBit - Adam.
    pbit_shift = series(tuned, SIGN_NOISY, "pbit") - series(tuned, SIGN_CLEAN, "pbit")
    adam_shift = series(tuned, SIGN_NOISY, "adam") - series(tuned, SIGN_CLEAN, "adam")
    effect = pbit_shift - adam_shift
    power = subsample(effect, sizes, args.draws, seed=11)

    print("sign-noise effect (PBit shift - Adam shift), true mean "
          f"{effect.mean():.3f}, sd {effect.std(ddof=1):.3f}")
    print(f"{'n':>4}{'P(negative)':>13}{'P(positive)':>13}{'P(sig @0.05)':>14}")
    for r in power:
        print(f"{r['n']:>4}{r['p_negative']:>13.3f}{r['p_positive']:>13.3f}"
              f"{r['p_significant_at_0.05']:>14.3f}")

    # Size check on the null: quant4 - clean for PBit (effect is exactly zero).
    null = series(tuned, QUANT_NOISY, "pbit") - series(tuned, SIGN_CLEAN, "pbit")
    null_power = subsample(null, sizes, args.draws, seed=13)

    print(f"\nquant4 null (PBit), true mean {null.mean():.2e}")
    print(f"{'n':>4}{'P(negative)':>13}{'P(positive)':>13}{'P(sig @0.05)':>14}")
    for r in null_power:
        print(f"{r['n']:>4}{r['p_negative']:>13.3f}{r['p_positive']:>13.3f}"
              f"{r['p_significant_at_0.05']:>14.3f}")

    payload = {
        "metadata": {
            "draws": args.draws,
            "source": args.tuned,
            "sizes": sizes,
        },
        "sign_noise_effect": {
            "mean": float(effect.mean()),
            "sd": float(effect.std(ddof=1)),
            "power_curve": power,
        },
        "quant4_null": {
            "mean": float(null.mean()),
            "power_curve": null_power,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
