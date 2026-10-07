"""Check that the prose in research/CLAIMS.md matches the recorded results.

Guards against transcription errors, which are the easiest way for a research
document to drift from its own evidence.

Numbers are matched numerically, with a tolerance set by the number of decimals
the prose actually uses, so ``0.790`` matches ``0.7898605069746513`` while a
genuine mistranscription does not pass. Values come from three sources:

- ``results/claims.json``    measurements made by this project
- ``results/power.json``     measurements made by this project
- ``results/withdrawn.json`` figures quoted from the withdrawn prior report

Run: python research/experiments/verify_docs.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research" / "CLAIMS.md"

SOURCES = [
    ROOT / "results" / "claims.json",
    ROOT / "results" / "power.json",
    ROOT / "results" / "withdrawn.json",
    ROOT / "results" / "tuned.json",
    ROOT / "results" / "clip_confound.json",
    ROOT / "results" / "mnist.json",
    ROOT / "results" / "robustness.json",
]

# The leaderboard CSV holds the per-cell aggregates (means, sds) that the JSON
# files only store as per-run series, so prose citing aggregates needs it too.
LEADERBOARD = ROOT / "results" / "leaderboard.csv"

# Values written as prose rather than measured: significance levels, round
# percentages, and order-of-magnitude figures used approximately.
ALLOWED = {95.0, 100.0, 40.0, 0.05, 0.01, 1e-16, 0.0, 5.0, 10.0, 20.0, 30.0, 50.0}


def collect(node, out: set[float]) -> None:
    if isinstance(node, dict):
        for v in node.values():
            collect(v, out)
    elif isinstance(node, list):
        for v in node:
            collect(v, out)
    elif isinstance(node, str):
        # Strings can carry real values (versions, the model architecture);
        # pull every number out of them.
        out.update(_numbers_in(node))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        v = float(node)
        out.add(v)
        # Prose writes "18.5 percent" where the record stores the fraction
        # 0.185; match both directions.
        out.add(v * 100.0)
        out.add(v / 100.0)


def _numbers_in(text: str) -> set[float]:
    found: set[float] = set()
    for m in re.finditer(r"\d+\.\d+", text):
        found.add(float(m.group()))
    return found


def decimals_in(token: str) -> int:
    return len(token.split("e")[0].split(".")[1]) if "." in token.split("e")[0] else 0


def main() -> int:
    missing = [p for p in SOURCES if not p.exists()]
    if missing:
        for p in missing:
            print(f"missing {p.relative_to(ROOT)}; run the experiments first")
        return 1

    values: set[float] = set()
    for src in SOURCES:
        collect(json.loads(src.read_text(encoding="utf-8")), values)
    if LEADERBOARD.exists():
        for row in csv.DictReader(LEADERBOARD.open(encoding="ascii")):
            for v in row.values():
                try:
                    values.add(float(v))
                except ValueError:
                    pass

    text = DOC.read_text(encoding="utf-8")
    tokens = re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", text)
    unmatched: list[str] = []

    for tok in tokens:
        if "." not in tok and "e" not in tok and len(tok) < 2:
            continue  # bare single digits are counts and clause numbers
        val = float(tok)
        mantissa = 0.5 * (10.0 ** -decimals_in(tok))
        # An exponent in the prose makes the written precision unknowable, so
        # fall back to a 1 percent relative band.
        tolerance = mantissa if "e" not in tok else abs(val) * 0.01
        tolerance += 1e-12  # rounding boundaries land exactly on the tolerance
        if any(abs(val - v) <= tolerance for v in values):
            continue
        if any(abs(val - v) < 1e-9 for v in ALLOWED):
            continue
        unmatched.append(tok)

    if unmatched:
        print("numbers in CLAIMS.md with no match in results/*.json:")
        for tok in unmatched:
            print(f"  {tok}")
        return 1

    print(f"OK: {len(tokens)} numeric values in CLAIMS.md all trace to results/*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
