"""Tests for the paired statistics helpers.

These lock the behaviour that the research claims rest on: that pairing is
actually available, that the permutation test is calibrated under the null, and
that multiplicity correction does what it claims.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))

import numpy as np
import pytest

from pbit.bench.stats import (
    Comparison,
    bootstrap_ci,
    describe,
    holm_bonferroni,
    paired_compare,
    sign_flip_p,
)


def test_paired_compare_detects_a_real_shift():
    rng = np.random.default_rng(0)
    b = rng.normal(10.0, 1.0, 40)
    a = b - 2.0  # paired: a is better by a constant
    c = paired_compare(a, b, label="shift")
    assert c.mean_diff == pytest.approx(-2.0, abs=1e-9)
    assert c.ci_hi < 0.0
    assert c.p_value < 0.01
    assert c.significant
    assert c.direction == "a better"


def test_paired_compare_finds_nothing_in_pure_noise():
    rng = np.random.default_rng(1)
    b = rng.normal(5.0, 3.0, 60)
    a = b + rng.normal(0.0, 3.0, 60)  # no systematic difference
    c = paired_compare(a, b)
    assert not c.significant or c.p_value > 0.05


def test_pairing_beats_unpaired_on_shared_start_variance():
    """Pairing is the whole reason this module exists.

    Two conditions share a large per-run offset (the start position). An
    unpaired difference of means has huge variance; the paired difference has
    almost none. The test asserts the paired statistic is far tighter.
    """
    rng = np.random.default_rng(2)
    shared = rng.normal(0.0, 50.0, 200)  # start-position effect
    effect = rng.normal(-1.0, 0.5, 200)  # small real improvement
    a = shared + effect
    b = shared

    paired_sd = float(np.std(a - b, ddof=1))
    # Standard error of the difference if the two series were compared unpaired.
    unpaired_se = float(
        np.sqrt(np.var(a, ddof=1) / a.size + np.var(b, ddof=1) / b.size)
    )
    assert unpaired_se > 4.0  # the shared offset dominates
    assert paired_sd < unpaired_se / 5


def test_sign_flip_p_is_calibrated_under_the_null():
    """Monte-Carlo p-values must not be systematically too small."""
    rng = np.random.default_rng(3)
    pvals = []
    for _ in range(200):
        d = rng.normal(0.0, 1.0, 25)
        pvals.append(sign_flip_p(d, n_perm=2000))
    # Under a true null, uniform p-values reject at about the nominal rate.
    assert 0.01 < float(np.mean(np.array(pvals) < 0.05)) < 0.12


def test_sign_flip_p_never_returns_zero():
    d = np.full(10, -5.0)
    p = sign_flip_p(d, n_perm=500)
    assert p > 0.0
    assert p <= 1.0


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(4)
    v = rng.normal(10.0, 2.0, 300)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi
    assert hi - lo > 0


def test_holm_is_monotone_and_capped():
    base = [
        Comparison("a", 10, 1, 1, -1, -2, -0.5, 0.001, 1),
        Comparison("b", 10, 1, 1, -1, -2, -0.5, 0.02, 1),
        Comparison("c", 10, 1, 1, -1, -2, -0.5, 0.30, 1),
    ]
    adj = holm_bonferroni(base, alpha=0.05)
    assert adj[0].p_value == pytest.approx(0.003)
    assert adj[1].p_value == pytest.approx(0.04)
    assert adj[2].p_value == pytest.approx(0.30)
    assert [c.p_value for c in adj] == sorted(c.p_value for c in adj)
    assert all(c.p_value <= 1.0 for c in adj)


def test_holm_controls_family_wise_error_under_the_null():
    rng = np.random.default_rng(5)
    comps = [
        paired_compare(rng.normal(0, 1, 20), rng.normal(0, 1, 20), label=f"null{i}")
        for i in range(20)
    ]
    adj = holm_bonferroni(comps, alpha=0.05)
    rejections = sum(1 for c in adj if c.p_value < 0.05)
    assert rejections <= 1


def test_paired_compare_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        paired_compare(np.zeros(5), np.zeros(4))


def test_describe_reports_spread():
    d = describe(np.array([1.0, 2.0, 3.0, 4.0]), label="x")
    assert d["n"] == 4
    assert d["mean"] == pytest.approx(2.5)
    assert d["sd"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert d["min"] == 1.0


def test_claim_document_numbers_trace_to_recorded_results():
    """Every number in CLAIMS.md must come from a results file.

    This caught two hand-computed Holm-adjusted p-values in the prose that did
    not match the values the script recorded.
    """
    from verify_docs import main as verify

    assert verify() == 0
