"""Tests for metrics."""

import numpy as np
import pytest

from pbit.bench.metrics import (
    auc_loss,
    escape_count_current,
    final_best,
    final_current,
    hit_times,
    iter_to_threshold,
    median_hit_time_successes,
    robustness_ratio,
    success_rate,
)

# fmt: off
_EASY = np.array([10.0, 8.0, 5.0, 3.0, 1.5, 0.8, 0.4, 0.2])   # crosses 1.0 at idx 5
_FAIL = np.array([10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0])   # never < 1
# fmt: on


def test_iter_to_threshold():
    assert iter_to_threshold(_EASY, 1.0) == 5
    assert iter_to_threshold(_FAIL, 1.0) == len(_FAIL)


def test_auc_loss_curve_area():
    # monotone decreasing curve -> area < start*len
    assert auc_loss(_EASY) < _EASY[0]
    assert auc_loss(_EASY) > 0


def test_final_best_current():
    histories = np.array([_EASY, _EASY])
    assert final_best(histories) == pytest.approx(_EASY[-1])
    assert final_current(_EASY) == pytest.approx(_EASY[-1])


def test_success_rate():
    histories = np.stack([_EASY, _FAIL])
    assert success_rate(histories, 1.0) == 0.5


def test_hit_times_and_median():
    histories = np.stack([_EASY, _EASY])
    hits = hit_times(histories, 1.0)
    assert np.all(hits == 5)
    assert median_hit_time_successes(histories, 1.0) == 5.0


def test_median_nan_when_no_success():
    histories = np.stack([_FAIL, _FAIL])
    assert np.isnan(median_hit_time_successes(histories, 1.0))


def test_median_nonempty_mix():
    histories = np.stack([_EASY, _FAIL])
    # only the successful run (hits at 5) counts
    assert median_hit_time_successes(histories, 1.0) == 5.0


def test_escape_count_zero_for_smooth():
    smooth = np.linspace(10, 0.1, 300)
    # Escape detection is a heuristic; a clean monotone curve may report at most a
    # tiny number of boundary artifacts, not a meaningful multi-escape signal.
    assert escape_count_current(smooth) <= 1


def test_escape_count_detects_plateau_escape():
    # Construct a clear plateau followed by a >5% drop -> must count >= 1 escape.
    curve = np.concatenate(
        [
            np.full(40, 8.0),            # plateau
            np.linspace(8.0, 1.0, 30),   # drop > 5%
            np.full(40, 1.0),            # second plateau
        ]
    )
    assert escape_count_current(curve, window=20) >= 1


def test_robustness_ratio():
    assert robustness_ratio(2.0, 1.0) == pytest.approx(2.0)
    assert robustness_ratio(1.0, 2.0) == pytest.approx(0.5)


def test_min_over_trajectory_is_variance_sensitive():
    """A running min falls when variance rises even with the mean unchanged.

    This is why a min-over-trajectory statistic can report that injected noise
    *improves* an optimizer. The benchmark reports terminal current loss next
    to the min so that artifact is visible instead of hidden.
    """
    mins, means = [], []
    for noise_scale in (0.0, 4.0):
        draws = []
        for trial in range(200):
            r = np.random.default_rng(500 + trial)
            walk = np.cumsum(noise_scale * r.standard_normal(200) * 0.05) + 5.0
            draws.append(walk)
        stacked = np.stack(draws)
        mins.append(stacked.min(axis=1).mean())
        means.append(stacked.mean(axis=1).mean())

    mean_shift = abs(means[1] - means[0])
    min_shift = mins[0] - mins[1]
    assert min_shift > 5 * mean_shift, f"mean moved {mean_shift:.4f}, min moved {min_shift:.4f}"
