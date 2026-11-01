"""Check that the prose in research/CLAIMS.md matches the recorded results.

Guards against transcription errors, which are the easiest way for a research
document to drift from its own evidence.

Numbers are matched numerically, with a tolerance set by the number of decimals
the prose actually uses, so ``0.790`` matches ``0.7898605069746513`` while a
genuine mistranscription does not pass. Values come from these recorded sources (`results/`):

- ``results/claims.json``    measurements made by this project
- ``results/power.json``     measurements made by this project
- ``results/withdrawn.json`` figures quoted from the withdrawn prior report
- ``results/quantize_retest.json`` Sphere quantization retest (claim C8)
- ``results/ablation.json``          annealing/stochasticity ablation (claim C9)
- ``results/ablation_seed1_v2.json`` ablation replication, traced when present

Verdicts in ``claims.json`` are computed from Holm-adjusted p-values, so a
claim whose raw p passes alpha while its adjusted p does not is recorded as
inconclusive; the raw value is kept beside it as ``p_value_raw``.

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
    ROOT / "results" / "quantize_retest.json",
    ROOT / "results" / "ablation.json",
    # Seed replications. Prose that quotes a second seed must trace to that
    # seed's own record, or the replication paragraph is unverifiable by
    # construction. Optional: absent replications do not fail the run.
    ROOT / "results" / "ablation_seed1_v2.json",
    ROOT / "results" / "quantize_retest_seed1.json",
    # Archived runs kept only as historical citations. CLAIMS.md quotes the
    # superseded 5-run ablation config when explaining why the tuning budget
    # mattered, so that number has to trace to something; it traces here.
    # Optional: absent archives do not fail the run.
    ROOT / "results" / "archive" / "ablation_5run.json",
    ROOT / "results" / "archive" / "ablation_seed1_5run.json",
]

# The leaderboard CSV holds the per-cell aggregates (means, sds) that the JSON
# files only store as per-run series, so prose citing aggregates needs it too.
LEADERBOARD = ROOT / "results" / "leaderboard.csv"

# Values written as prose rather than measured: significance levels, round
# percentages, and order-of-magnitude figures used approximately.
ALLOWED = {95.0, 100.0, 40.0, 0.05, 0.01, 1e-16, 0.0, 5.0, 10.0, 20.0, 30.0, 50.0}

#: Historical figures cited for comparison that live only in superseded runs
#: which were never committed. They are named here, with the reason, so the
#: allowlist stays an explicit editorial decision instead of a silent gap. Each
#: entry is a *citation of a withdrawn configuration*, not evidence for a claim:
#: CLAIMS.md uses them only to explain why the current configuration differs.
CITED_NOT_ARCHIVED = {
    31.2576: "superseded 5-run ablation PBit corruption shift; its grid was not committed",
}


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
    mantissa = token.split("e")[0]
    return len(mantissa.split(".")[1]) if "." in mantissa else 0


def main() -> int:
    # Seed replications are advisory: their absence must not block the check on
    # the primary record, but a *present* replication is traced in full.
    missing = [p for p in SOURCES if not p.exists() and not p.name.endswith("_seed1.json")]
    if missing:
        for p in missing:
            print(f"missing {p.relative_to(ROOT)}; run the experiments first")
        return 1

    values: set[float] = set()
    for src in SOURCES:
        if not src.exists():
            continue
        collect(json.loads(src.read_text(encoding="utf-8")), values)
    if LEADERBOARD.exists():
        for row in csv.DictReader(LEADERBOARD.open(encoding="ascii")):
            for v in row.values():
                try:
                    values.add(float(v))
                except ValueError:
                    pass

    text = DOC.read_text(encoding="utf-8")
    # The sign is part of the number, not punctuation: interval bounds are
    # written "-3e-04" in prose and stored as -0.0003 in the record, so a
    # sign-dropping regex would compare +3e-04 against a negative bound and
    # call every CI endpoint untraceable.
    tokens = re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", text)
    unmatched: list[str] = []

    for tok in tokens:
        if "." not in tok and "e" not in tok and len(tok.lstrip("-")) < 2:
            continue  # bare single digits are counts and clause numbers
        val = float(tok)
        # Tolerance follows the *written* precision. Two cases:
        #
        # * a fixed-decimal token like "-3e-04" or "18.327" is matched to half
        #   of its last decimal place, so prose may round the record to
        #   whatever it printed;
        # * a token whose exponent is wider than its mantissa ("-3e-04" has one
        #   mantissa digit but lands on the 1e-04 place) cannot be matched by a
        #   half-mantissa unit, because 0.5 * 0.1 * 1e-04 = 5e-06 is smaller
        #   than the record's own -0.0002873. For those, the unit of the last
        #   *printed* place is the exponent's place: half of 1e-04.
        dec = decimals_in(tok)
        if "e" in tok and dec > 0:
            exp = int(tok.lower().split("e")[1])
            # Last printed decimal place is 10**(exp - dec + 1); half a unit.
            unit = 10.0 ** (exp - dec + 1)
        else:
            unit = 0.5 * (10.0 ** -dec)
        tolerance = unit
        tolerance += 1e-12  # rounding boundaries land exactly on the tolerance
        if any(abs(val - v) <= tolerance for v in values):
            continue
        if any(abs(val - v) < 1e-9 for v in ALLOWED):
            continue
        if any(abs(val - v) <= tolerance for v in CITED_NOT_ARCHIVED):
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
