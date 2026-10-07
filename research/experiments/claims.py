"""Claim battery over the tuned, held-out evaluation series.

Reads ``results/tuned.json`` (produced by ``tune_and_evaluate.py``) and computes
every comparison from the recorded per-run series. No optimization is re-run
here: the numbers in this file's output are derived only from recorded data,
and any re-run of the whole battery is bit-reproducible from the same tuned
input.

Every claim is paired. Run ``i`` for optimizer A and run ``i`` for optimizer B
in the same cell share a start position, a noise perturbation, and (for the
same optimizer across noise conditions) the same private coins, so differencing
within a run cancels all three.

Verdicts are assigned from the bootstrap interval and the sign-flip p-value,
and only when they agree. All claims form one family and get a Holm correction.

Usage:
    python research/experiments/claims.py
    python research/experiments/claims.py --tuned results/tuned.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pbit.bench.stats import (  # noqa: E402
    Comparison,
    describe,
    holm_bonferroni,
    paired_compare,
)

ALPHA = 0.05
ITERS_SENTINEL = 300  # hit_time == max_iter means the threshold was never hit

CELL = {
    "rast_d2_clean": "Rastrigin/dim=2/clean",
    "rast_d2_gauss": "Rastrigin/dim=2/GaussianNoise_sigma=0.5",
    "rast_d2_corrupt": "Rastrigin/dim=2/CorruptionNoise_p=0.2",
    "rast_d2_quant4": "Rastrigin/dim=2/QuantizeNoise_bits=4_stochastic=True",
    "rast_d2_sign": "Rastrigin/dim=2/SignNoise",
    "rast_d10_clean": "Rastrigin/dim=10/clean",
    "ack_d2_clean": "Ackley/dim=2/clean",
    "ack_d2_gauss": "Ackley/dim=2/GaussianNoise_sigma=0.5",
    "ack_d2_corrupt": "Ackley/dim=2/CorruptionNoise_p=0.2",
    "ack_d2_quant4": "Ackley/dim=2/QuantizeNoise_bits=4_stochastic=True",
    "ack_d2_sign": "Ackley/dim=2/SignNoise",
    "rosen_d2_clean": "Rosenbrock/dim=2/clean",
    "rosen_d2_gauss": "Rosenbrock/dim=2/GaussianNoise_sigma=0.5",
    "rosen_d2_corrupt": "Rosenbrock/dim=2/CorruptionNoise_p=0.2",
    "rosen_d2_quant4": "Rosenbrock/dim=2/QuantizeNoise_bits=4_stochastic=True",
    "rosen_d2_sign": "Rosenbrock/dim=2/SignNoise",
    "rosen_d10_clean": "Rosenbrock/dim=10/clean",
}

NOISES = [
    ("gaussian", "gauss"),
    ("corruption", "corrupt"),
    ("quant4", "quant4"),
    ("sign", "sign"),
]

LANDSCAPES = ["rast", "ack", "rosen"]

LANDSCAPE_NAMES = {"rast": "Rastrigin", "ack": "Ackley", "rosen": "Rosenbrock"}


def load_series(tuned: dict, cell_key: str, optimizer: str, metric: str) -> np.ndarray:
    """Per-run series for one optimizer in one cell, ordered by run id."""
    per_run = tuned["evaluation"][cell_key][optimizer][metric]
    return np.array([per_run[k] for k in sorted(per_run, key=int)], dtype=np.float64)


def _verdict(cmp: Comparison) -> str:
    """Assign a verdict only when the interval and the permutation test agree."""
    ci_excludes_zero = cmp.ci_lo > 0.0 or cmp.ci_hi < 0.0
    test_rejects = cmp.p_value < ALPHA
    detail = f"CI [{cmp.ci_lo:.3f}, {cmp.ci_hi:.3f}], p={cmp.p_value:.4f}"
    if ci_excludes_zero != test_rejects:
        return f"INCONSISTENT: interval and test disagree, {detail}"
    if not ci_excludes_zero:
        return f"UNRESOLVED: no detectable difference, {detail}"
    side = "lower" if cmp.mean_diff < 0 else "higher"
    return f"RESOLVED: {side}, {detail}"


def _supports(cmp: Comparison, expected: str) -> str:
    """State whether the measurement agrees with the registered hypothesis."""
    if cmp.p_value >= ALPHA or not (cmp.ci_lo > 0.0 or cmp.ci_hi < 0.0):
        return "INCONCLUSIVE: no detectable effect"
    if expected == "no_better":
        return "CONFIRMED" if cmp.mean_diff >= 0 else "REFUTED: noise improved terminal loss"
    if expected == "a_lower":
        return "CONFIRMED" if cmp.mean_diff < 0 else "REFUTED: not lower"
    if expected == "a_shift_less":
        return "CONFIRMED" if cmp.mean_diff < 0 else "REFUTED: shifted at least as much"
    raise ValueError(f"unknown expectation {expected!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", type=str, default="results/tuned.json")
    ap.add_argument("--out", type=str, default="results/claims.json")
    args = ap.parse_args()

    tuned_path = Path(args.tuned)
    if not tuned_path.exists():
        raise SystemExit(f"missing {tuned_path}; run tune_and_evaluate.py first")
    tuned = json.loads(tuned_path.read_text(encoding="utf-8"))
    n_runs = tuned["metadata"]["eval_runs"]

    claims: list[dict] = []
    holm_targets: list[dict] = []
    comparators: list[Comparison] = []

    def register(cid: str, question: str, statistic: str, hypothesis: str,
                 expected: str, cmp: Comparison, extra: dict | None = None):
        entry = {
            "cid": cid,
            "question": question,
            "statistic": statistic,
            "hypothesis": hypothesis,
            "comparison": cmp.to_dict(),
            "verdict": _verdict(cmp),
            "supports_hypothesis": _supports(cmp, expected),
        }
        if extra:
            entry.update(extra)
        claims.append(entry)
        holm_targets.append(entry["comparison"])
        comparators.append(cmp)
        print(f"--- {cid}: {question}")
        print(f"    {entry['verdict']}")
        print(f"    hypothesis: {entry['supports_hypothesis']}")

    # C1: does 4-bit stochastic quantization change tuned PBit's terminal loss?
    clean = load_series(tuned, CELL["rast_d2_clean"], "pbit", "final_current")
    quant = load_series(tuned, CELL["rast_d2_quant4"], "pbit", "final_current")
    register(
        "C1",
        "Does 4-bit stochastic quantization change PBit's terminal loss (tuned)?",
        "paired 4-bit-stochastic vs clean, terminal current loss, tuned PBit, Rastrigin dim=2",
        "noise cannot make the terminal loss lower than clean",
        "no_better",
        paired_compare(quant, clean, "quant4 vs clean, tuned PBit"),
        {"clean": describe(clean, "clean"), "noisy": describe(quant, "quant4")},
    )

    # C2: sign-noise degradation, tuned PBit vs tuned Adam.
    def shift(cell_noisy: str, cell_clean: str, opt: str) -> np.ndarray:
        noisy = load_series(tuned, cell_noisy, opt, "final_current")
        base = load_series(tuned, cell_clean, opt, "final_current")
        return noisy - base

    pbit_shift = shift(CELL["rast_d2_sign"], CELL["rast_d2_clean"], "pbit")
    adam_shift = shift(CELL["rast_d2_sign"], CELL["rast_d2_clean"], "adam")
    register(
        "C2",
        "Is tuned PBit less degraded by 1-bit sign-only gradients than tuned Adam?",
        "paired comparison of noisy-clean shift, PBit vs Adam, Rastrigin dim=2",
        "PBit shifts less than Adam",
        "a_shift_less",
        paired_compare(pbit_shift, adam_shift, "sign-noise shift, PBit vs Adam"),
        {"pbit_shift": describe(pbit_shift, "PBit shift"),
         "adam_shift": describe(adam_shift, "Adam shift")},
    )

    # C3: tuned PBit vs tuned Adam on Rosenbrock, both dimensions.
    for cid, cell_key, dim in (("C3a", CELL["rosen_d2_clean"], 2), ("C3b", CELL["rosen_d10_clean"], 10)):
        pb = load_series(tuned, cell_key, "pbit", "final_current")
        ad = load_series(tuned, cell_key, "adam", "final_current")
        register(
            cid,
            f"Does tuned PBit reach lower terminal loss than tuned Adam on Rosenbrock dim={dim}?",
            f"paired PBit vs Adam, terminal current loss, Rosenbrock dim={dim}, clean",
            "PBit lower",
            "a_lower",
            paired_compare(pb, ad, f"PBit vs Adam, Rosenbrock dim={dim}"),
            {"pbit": describe(pb, "PBit"), "adam": describe(ad, "Adam")},
        )

    # C4: threshold speed with failure counts (Rosenbrock dim=2).
    pb_hits = load_series(tuned, CELL["rosen_d2_clean"], "pbit", "hit_time")
    ad_hits = load_series(tuned, CELL["rosen_d2_clean"], "adam", "hit_time")
    pb_fail = int(np.sum(pb_hits >= ITERS_SENTINEL))
    ad_fail = int(np.sum(ad_hits >= ITERS_SENTINEL))
    both = (pb_hits < ITERS_SENTINEL) & (ad_hits < ITERS_SENTINEL)
    if both.sum() >= 2:
        cmp = paired_compare(pb_hits[both], ad_hits[both], "hit time, both succeeded")
        c4_verdict = _verdict(cmp)
        c4_support = _supports(cmp, "a_lower")
        c4_cmp = cmp.to_dict()
    else:
        c4_verdict = "INSUFFICIENT DATA: too few runs where both optimizers succeeded"
        c4_support = "INCONCLUSIVE"
        c4_cmp = None
    c4 = {
        "cid": "C4",
        "question": "Does tuned PBit reach the success threshold sooner than tuned Adam on Rosenbrock?",
        "statistic": "paired iterations to loss < 1.0, runs where both succeeded; failures reported",
        "hypothesis": "PBit needs fewer iterations",
        "comparison": c4_cmp,
        "pbit_failures": pb_fail,
        "adam_failures": ad_fail,
        "n_runs": n_runs,
        "both_succeeded": int(both.sum()),
        "pbit_hit_times": describe(pb_hits[pb_hits < ITERS_SENTINEL], "PBit hit time"),
        "adam_hit_times": describe(ad_hits[ad_hits < ITERS_SENTINEL], "Adam hit time"),
        "verdict": c4_verdict,
        "supports_hypothesis": c4_support,
    }
    claims.append(c4)
    if c4_cmp is not None:
        holm_targets.append(c4["comparison"])
        comparators.append(
            Comparison(**{k: c4_cmp[k] for k in
                          ("label", "n", "mean_a", "mean_b", "mean_diff",
                           "ci_lo", "ci_hi", "p_value", "sd_diff")})
        )
    print(f"--- C4: {c4['question']}")
    print(f"    {c4['verdict']} (failures: pbit={pb_fail}, adam={ad_fail})")

    # C5: does the min statistic disagree with the terminal statistic?
    readings = {}
    for reader, metric in (("min_statistic", "final_best"),
                           ("terminal_statistic", "final_current")):
        base = load_series(tuned, CELL["rast_d2_clean"], "pbit", metric)
        noisy = load_series(tuned, CELL["rast_d2_quant4"], "pbit", metric)
        readings[reader] = paired_compare(noisy, base, f"quant4-clean via {reader}")
    m, t = readings["min_statistic"], readings["terminal_statistic"]
    # A sign comparison is meaningless at machine-epsilon scale: with a true
    # effect of zero, the two readers' mean differences are ~1e-16 and their
    # signs are noise. Treat differences below this floor as zero.
    scale = max(1.0, abs(m.mean_a), abs(m.mean_b), abs(t.mean_a), abs(t.mean_b))
    floor = 1e-9 * scale
    m_zero = abs(m.mean_diff) < floor
    t_zero = abs(t.mean_diff) < floor
    disagree_sign = (not m_zero and not t_zero) and ((m.mean_diff < 0) != (t.mean_diff < 0))
    disagree_sig = m.significant != t.significant
    disagree = disagree_sign or disagree_sig
    claims.append({
        "cid": "C5",
        "question": "Can a min-over-trajectory statistic reverse the sign of a noise effect?",
        "statistic": "the same paired comparison read by min statistic and by terminal statistic",
        "hypothesis": "the two readers can disagree on sign or significance",
        "min_statistic": m.to_dict(),
        "terminal_statistic": t.to_dict(),
        "readings_disagree": bool(disagree),
        "verdict": ("SUPPORTED: the two readers disagree" if disagree
                    else "NOT SUPPORTED: both readers agree"),
        "supports_hypothesis": "CONFIRMED" if disagree else "REFUTED: readers agree",
    })
    print(f"--- C5: {claims[-1]['question']}")
    print(f"    {claims[-1]['verdict']}")

    # C6: per-noise-family degradation across all three landscapes (dim=2).
    # The corruption effect must replicate on more than one landscape to count.
    sub = []
    for ls in LANDSCAPES:
        for label, suffix in NOISES:
            ps = shift(CELL[f"{ls}_d2_{suffix}"], CELL[f"{ls}_d2_clean"], "pbit")
            ads = shift(CELL[f"{ls}_d2_{suffix}"], CELL[f"{ls}_d2_clean"], "adam")
            cmp = paired_compare(ps, ads, f"{label} shift, PBit vs Adam, {LANDSCAPE_NAMES[ls]}")
            sub.append({
                "landscape": LANDSCAPE_NAMES[ls],
                "noise": label,
                "comparison": cmp.to_dict(),
                "pbit_shift": describe(ps, "PBit shift"),
                "adam_shift": describe(ads, "Adam shift"),
                "verdict": _verdict(cmp),
                "supports_hypothesis": _supports(cmp, "a_shift_less"),
            })
            holm_targets.append(sub[-1]["comparison"])
            comparators.append(cmp)
    claims.append({
        "cid": "C6",
        "question": "Across noise families and landscapes, is tuned PBit degraded less than tuned Adam?",
        "statistic": "paired noisy-clean shift per (landscape, noise), PBit vs Adam, dim=2",
        "hypothesis": "PBit shifts less than Adam in each family",
        "per_noise": sub,
        "verdict": "see per-noise entries",
    })
    print("--- C6: per-noise degradation across landscapes")
    for s in sub:
        print(f"    {s['landscape']:<11} {s['noise']:<11} {s['verdict']}")

    # C7: does the PBit-vs-Adam deficit grow with dimension (Rastrigin, clean)?
    gap_d2 = (load_series(tuned, CELL["rast_d2_clean"], "pbit", "final_current")
              - load_series(tuned, CELL["rast_d2_clean"], "adam", "final_current"))
    gap_d10 = (load_series(tuned, CELL["rast_d10_clean"], "pbit", "final_current")
               - load_series(tuned, CELL["rast_d10_clean"], "adam", "final_current"))
    register(
        "C7",
        "Does PBit's deficit against Adam grow from dim=2 to dim=10 (Rastrigin, clean)?",
        "paired (PBit-Adam) gaps at dim=2 vs dim=10; run i shares the init-stream prefix across dims",
        "the deficit grows with dimension",
        "a_lower",
        paired_compare(gap_d2, gap_d10, "PBit-Adam gap, dim2 vs dim10"),
        {"gap_d2": describe(gap_d2, "gap dim=2"), "gap_d10": describe(gap_d10, "gap dim=10")},
    )

    # One family, one correction.
    for holder, adj in zip(holm_targets, holm_bonferroni(comparators)):
        holder["p_value_holm"] = adj.p_value

    # Leaderboard: tuned mean terminal loss with sd, every optimizer, every cell.
    lb_path = Path("results/leaderboard.csv")
    with lb_path.open("w", newline="", encoding="ascii") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "optimizer", "selected_params", "n_runs",
                    "final_current_mean", "final_current_sd", "final_best_mean",
                    "success_rate"])
        for cell_key, per_opt in tuned["evaluation"].items():
            for opt_name, metrics in per_opt.items():
                fc = np.array([metrics["final_current"][k]
                               for k in sorted(metrics["final_current"], key=int)])
                fb = np.array([metrics["final_best"][k]
                               for k in sorted(metrics["final_best"], key=int)])
                ht = np.array([metrics["hit_time"][k]
                               for k in sorted(metrics["hit_time"], key=int)])
                selected = tuned["eval_configs"][cell_key][opt_name]
                w.writerow([cell_key, opt_name, json.dumps(selected), len(fc),
                            round(float(fc.mean()), 6), round(float(fc.std(ddof=1)), 6),
                            round(float(fb.mean()), 6),
                            round(float(np.mean(ht < ITERS_SENTINEL)), 4)])

    payload = {
        "metadata": {**tuned["metadata"], "source": str(tuned_path)},
        "results": claims,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"wrote {lb_path}")


if __name__ == "__main__":
    main()
