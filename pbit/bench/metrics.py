"""Metrics for stochastic optimizer benchmarking.

All *algorithmic* metrics operate on **best-so-far** histories where noted, not
the noisy current iterate. `iter_to_threshold` is never averaged over failed
runs; instead use ``success_rate`` and ``median_hit_time_successes``.

The key honesty rule: measured wall-time is separate from declared proxy FLOPs;
there is no single "efficiency" number that mixes the two.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Core curve metrics
# --------------------------------------------------------------------------


def iter_to_threshold(best_history: np.ndarray, threshold: float) -> int:
    """First index where best-so-far loss < threshold, else ``len`` (never hit)."""
    best_history = np.asarray(best_history)
    if best_history.size == 0:
        raise ValueError("best_history must not be empty")
    idxs = np.where(best_history < threshold)[0]
    return int(idxs[0]) if len(idxs) > 0 else int(len(best_history))


def auc_loss(history: np.ndarray) -> float:
    """Normalized area under the (best-so-far) loss curve.

    Warning: scale-dependent; only meaningful for comparison *within* a single
    benchmark function, never across functions with different loss ranges.
    """
    history = np.asarray(history, dtype=np.float64)
    if history.size < 2:
        return 0.0
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return float(trapz(history) / len(history))


def final_best(histories: np.ndarray) -> float:
    """Mean of final best-so-far loss across runs."""
    return float(np.mean(np.asarray(histories)[:, -1]))


def final_current(histories: np.ndarray) -> float:
    """Mean of final current loss across runs (accepts 1-D or 2-D)."""
    a = np.asarray(histories)
    if a.ndim == 1:
        return float(a[-1])
    return float(np.mean(a[:, -1]))


# --------------------------------------------------------------------------
# Success-oriented metrics (threshold-based, robust to failed runs)
# --------------------------------------------------------------------------


def success_rate(best_histories: np.ndarray, threshold: float) -> float:
    """Fraction of runs whose best-so-far loss reaches ``threshold``."""
    best_histories = np.asarray(best_histories)
    success = np.any(best_histories < threshold, axis=1)
    return float(np.mean(success))


def hit_times(best_histories: np.ndarray, threshold: float) -> np.ndarray:
    """Iteration index at which each run first reaches ``threshold``.

    Runs that never reach threshold are assigned ``len`` (a sentinel for failure).
    """
    best_histories = np.asarray(best_histories)
    n_runs, n_iter = best_histories.shape
    hits = np.full(n_runs, n_iter, dtype=float)
    for r in range(n_runs):
        idxs = np.where(best_histories[r] < threshold)[0]
        if len(idxs) > 0:
            hits[r] = idxs[0]
    return hits


def median_hit_time_successes(best_histories: np.ndarray, threshold: float) -> float:
    """Median hit-time **only over successful runs**.

    Returns NaN if no run succeeds. Do not average over failed runs (that would
    conflate 'failed to reach' with 'took longer').
    """
    hits = hit_times(best_histories, threshold)
    successes = hits[hits < len(np.asarray(best_histories).T)]
    if successes.size == 0:
        return float("nan")
    return float(np.median(successes))


# --------------------------------------------------------------------------
# Exploration (basin-hopping) metric
# --------------------------------------------------------------------------


def escape_count_current(mean_current: np.ndarray, window: int = 20, delta_threshold: float = 0.05) -> int:
    """Count plateau-escapes on the *current*-loss mean curve.

    A plateau is a window whose coefficient of variation ``std/|mean|`` is small;
    an escape is logged when loss drops >5% after a plateau. Parameterized and
    documented as exploratory (not a headline metric).
    """
    mean_current = np.asarray(mean_current, dtype=np.float64)
    escapes = 0
    in_plateau = False
    for i in range(window, len(mean_current) - window):
        segment = mean_current[i - window : i]
        local_std = np.std(segment)
        local_mean = np.abs(np.mean(segment)) + 1e-12
        if local_std / local_mean < delta_threshold:
            in_plateau = True
        elif in_plateau:
            before = np.mean(mean_current[i - window : i])
            after = np.mean(mean_current[i : i + window])
            if before > 0 and (before - after) / before > 0.05:
                escapes += 1
            in_plateau = False
    return escapes


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


def robustness_ratio(noisy_final_best: float, clean_final_best: float) -> float:
    """``noisy / clean`` final best loss.

    ``1.0`` = no degradation; ``>1`` = degradation; ``<1`` = noisy gradients helped
    (possible for stochastic optimizers). Small epsilon guards division by zero.
    """
    return float(noisy_final_best / max(clean_final_best, 1e-12))


def quantize_sweep(
    run_fn,
    optimizers,
    function,
    bits_list,
    n_runs: int = 10,
    max_iter: int = 500,
    seed: int = 42,
) -> dict[str, dict[int, float]]:
    """Run a quantization bit-width sweep, returning ``{optimizer: {bits: final_best}}``.

    ``run_fn`` must accept (function, noise, optimizer, max_iter, n_runs, seed) and
    return something exposing ``final_best_loss()`` — used by examples to keep the
    sweep generic without importing the full runner here.
    """
    from pbit.bench.noise import QuantizeNoise

    results: dict[str, dict[int, float]] = {}
    for opt_name, opt in optimizers.items():
        results[opt_name] = {}
        for bits in bits_list:
            noise = QuantizeNoise(bits=bits, stochastic=True)
            report = run_fn(function, noise, opt, max_iter=max_iter, n_runs=n_runs, seed=seed)
            results[opt_name][bits] = report.final_best_loss()
    return results
