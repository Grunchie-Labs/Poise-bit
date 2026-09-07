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

__all__ = [
    "Ackley",
    "ClipNoise",
    "CorruptionNoise",
    "Experiment",
    "FunctionSpec",
    "GaussianNoise",
    "NoNoise",
    "NoiseSpec",
    "OptimizerSpec",
    "QuantizeNoise",
    "Rastrigin",
    "Report",
    "ReportDiff",
    "Rosenbrock",
    "SignNoise",
    "auc_loss",
    "compare_reports",
    "escape_count_current",
    "final_best",
    "get_function",
    "iter_to_threshold",
    "make_fitness",
    "median_hit_time_successes",
    "robustness_ratio",
    "success_rate",
]
