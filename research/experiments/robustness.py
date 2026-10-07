"""Check that every claim verdict is stable across master eval seeds.

The claim battery runs on 50 runs from one eval seed. A reviewer will ask
whether the verdicts survive a different seed, so the battery is run at several
eval seeds (tuning unchanged, since tuning uses its own seed) and this script
checks that every claim lands in the same category at every seed.

A verdict is stable when its category (resolved direction, unresolved,
inconsistent, supported/not-supported) matches across all seeds. Raw effect
sizes vary between seeds; the category is the claim.

Usage:
    python research/experiments/tune_and_evaluate.py --eval-seed 1 --out results/tuned_seed1.json
    python research/experiments/claims.py --tuned results/tuned_seed1.json --out results/claims_seed1.json
    ... (repeat per seed) ...
    python research/experiments/robustness.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def category(verdict: str) -> str:
    """The claim-level content of a verdict string, without its numbers."""
    if verdict.startswith("RESOLVED: lower"):
        return "resolved_lower"
    if verdict.startswith("RESOLVED: higher"):
        return "resolved_higher"
    if verdict.startswith("UNRESOLVED"):
        return "unresolved"
    if verdict.startswith("INCONSISTENT"):
        return "inconsistent"
    if verdict.startswith("SUPPORTED"):
        return "supported"
    if verdict.startswith("NOT SUPPORTED"):
        return "not_supported"
    if verdict.startswith("INSUFFICIENT"):
        return "insufficient_data"
    return "see per-noise entries"


def load(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for r in data["results"]:
        cmp = r.get("comparison") or {}
        out[r["cid"]] = {"verdict": r["verdict"], "mean_diff": cmp.get("mean_diff")}
        if r["cid"] == "C6":
            for s in r["per_noise"]:
                out[f"C6/{s['landscape']}/{s['noise']}"] = {
                    "verdict": s["verdict"],
                    "mean_diff": s["comparison"]["mean_diff"],
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=str, default="results/robustness.json")
    args = ap.parse_args()

    per_seed: dict[int, dict[str, str]] = {}
    for s in args.seeds:
        path = Path("results/claims.json") if s == 0 else Path(f"results/claims_seed{s}.json")
        if not path.exists():
            raise SystemExit(f"missing {path}; run tune_and_evaluate.py and claims.py for seed {s}")
        per_seed[s] = load(path)

    all_cids = sorted({cid for verdicts in per_seed.values() for cid in verdicts})
    rows = []
    n_stable = 0
    n_direction_stable = 0
    for cid in all_cids:
        cats = {s: category(per_seed[s][cid]["verdict"]) for s in args.seeds if cid in per_seed[s]}
        diffs = [per_seed[s][cid]["mean_diff"] for s in args.seeds
                 if cid in per_seed[s] and per_seed[s][cid]["mean_diff"] is not None]
        stable = len(set(cats.values())) == 1
        # Direction stability: every non-tiny mean difference has the same sign.
        signs = {d > 0 for d in diffs if abs(d) > 1e-9}
        direction_stable = len(signs) <= 1
        n_stable += stable
        n_direction_stable += direction_stable
        rows.append({
            "claim": cid,
            "verdicts_by_seed": cats,
            "mean_diffs_by_seed": {s: per_seed[s][cid]["mean_diff"]
                                   for s in args.seeds if cid in per_seed[s]},
            "stable": stable,
            "direction_stable": direction_stable,
        })

    print(f"{'claim':<32}" + "".join(f"{'seed ' + str(s):>16}" for s in args.seeds) + "stable")
    for r in rows:
        print(f"{r['claim']:<32}" + "".join(f"{r['verdicts_by_seed'].get(s, '-'):>16}" for s in args.seeds)
              + ("yes" if r["stable"] else "NO"))

    payload = {
        "metadata": {"seeds": args.seeds},
        "rows": rows,
        "n_claims": len(rows),
        "n_stable": n_stable,
        "n_direction_stable": n_direction_stable,
        "all_stable": n_stable == len(rows),
        "all_direction_stable": n_direction_stable == len(rows),
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n{n_stable}/{len(rows)} verdict categories stable across seeds {args.seeds}")
    print(f"{n_direction_stable}/{len(rows)} effect directions stable across seeds {args.seeds}")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
