"""Benchmarking: reproducible Experiment/Report, functions, noise, metrics."""

from pbit.bench.compare import ReportDiff, compare_reports
from pbit.bench.experiment import Experiment
from pbit.bench.fitness import make_fitness
from pbit.bench.functions import Ackley, Rastrigin, Rosenbrock, get_function
from pbit.bench.metrics import (
    auc_loss,
    escape_count_current,
    final_best,
    iter_to_threshold,
    median_hit_time_successes,
    robustness_ratio,
    success_rate,
)
from pbit.bench.noise import (
    ClipNoise,
    CorruptionNoise,
    GaussianNoise,
    NoNoise,
    QuantizeNoise,
    SignNoise,
)
from pbit.bench.report import Report
from pbit.bench.specs import FunctionSpec, NoiseSpec, OptimizerSpec
from pbit.bench.stats import (
    Comparison,
    bootstrap_ci,
    describe,
    holm_bonferroni,
    paired_compare,
    sign_flip_p,
)
from pbit.bench.tune import (
    GridPoint,
    TuningResult,
    assert_seeds_disjoint,
    cross,
    grid_from_values,
    tune_optimizer,
)

__all__ = [
    "Ackley",
    "ClipNoise",
    "Comparison",
    "CorruptionNoise",
    "Experiment",
    "FunctionSpec",
    "GaussianNoise",
    "GridPoint",
    "NoNoise",
    "NoiseSpec",
    "OptimizerSpec",
    "QuantizeNoise",
    "Rastrigin",
    "Report",
    "ReportDiff",
    "Rosenbrock",
    "SignNoise",
    "TuningResult",
    "assert_seeds_disjoint",
    "auc_loss",
    "bootstrap_ci",
    "compare_reports",
    "cross",
    "describe",
    "escape_count_current",
    "final_best",
    "get_function",
    "grid_from_values",
    "holm_bonferroni",
    "iter_to_threshold",
    "make_fitness",
    "median_hit_time_successes",
    "paired_compare",
    "robustness_ratio",
    "sign_flip_p",
    "success_rate",
    "tune_optimizer",
]
