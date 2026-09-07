"""Report: aggregated results, metrics, metadata, export, and comparison.

A ``Report`` holds per-run data plus aggregated summary metrics across runs for
each (function, noise, optimizer) cell. It is the single public artifact of a
benchmark run and can export to CSV (ASCII) and JSON (with full metadata).
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pbit import __version__
from pbit.bench import metrics as M


@dataclass
class _RunRow:
    function: str
    dim: int
    noise: str
    optimizer: str
    run_id: int
    current_history: np.ndarray
    best_history: np.ndarray
    step_times: np.ndarray
    wall_time_total: float
    clip_count: int
    diagnostics: dict[str, Any]
    flops_per_step: int | None


@dataclass
class _Cell:
    """Aggregated metrics for one (function, noise, optimizer) cell."""

    function: str
    dim: int
    noise: str
    optimizer: str
    n_runs: int
    success_rate: float = float("nan")
    median_hit_time_successes: float = float("nan")
    final_best_loss_mean: float = float("nan")
    final_best_loss_median: float = float("nan")
    final_current_loss_mean: float = float("nan")
    auc_best_loss: float = float("nan")
    auc_current_loss: float = float("nan")
    escape_count_current: int = 0
    wall_time_total_mean: float = float("nan")
    wall_time_step_mean: float = float("nan")
    proxy_flops_per_step: float = float("nan")
    clip_count_mean: float = float("nan")
    threshold: float = 1.0


class Report:
    """Benchmark results with aggregation, export, and comparison helpers."""

    def __init__(self, experiment: Any = None) -> None:
        self._experiment = experiment
        self._rows: list[_RunRow] = []
        self._cells: list[_Cell] | None = None
        self.threshold: float = (
            float(getattr(experiment, "config", {}).get("threshold", 1.0))
            if experiment is not None
            else 1.0
        )

    # ------------------------------------------------------------------ build
    def set_rows(self, rows: list[_RunRow]) -> None:
        self._rows = rows
        self._cells = None

    @property
    def rows(self) -> list[_RunRow]:
        """Public read-only access to raw run rows (for plotting / inspection)."""
        return list(self._rows)

    # ------------------------------------------------------------------ cells
    def cells(self) -> list[_Cell]:
        if self._cells is None:
            self._cells = self._aggregate()
        return self._cells

    def _aggregate(self) -> list[_Cell]:
        cells: dict[tuple[str, str, str], list[_RunRow]] = {}
        for r in self._rows:
            cells.setdefault((r.function, r.noise, r.optimizer), []).append(r)

        out: list[_Cell] = []
        for key, runs in cells.items():
            function, noise, optimizer = key
            dim = runs[0].dim
            best_all = np.stack([r.best_history for r in runs])  # (n_runs, iters)
            cur_all = np.stack([r.current_history for r in runs])
            thr = self.threshold

            cell = _Cell(
                function=function,
                dim=dim,
                noise=noise,
                optimizer=optimizer,
                n_runs=len(runs),
                success_rate=M.success_rate(best_all, thr),
                median_hit_time_successes=M.median_hit_time_successes(best_all, thr),
                final_best_loss_mean=M.final_best(best_all),
                final_best_loss_median=float(np.median(best_all[:, -1])),
                final_current_loss_mean=M.final_current(cur_all),
                auc_best_loss=M.auc_loss(np.mean(best_all, axis=0)),
                auc_current_loss=M.auc_loss(np.mean(cur_all, axis=0)),
                escape_count_current=M.escape_count_current(np.mean(cur_all, axis=0)),
                wall_time_total_mean=float(np.mean([r.wall_time_total for r in runs])),
                wall_time_step_mean=float(np.mean([np.mean(r.step_times) for r in runs])),
                proxy_flops_per_step=(
                    float(np.mean([r.flops_per_step for r in runs]))
                    if runs[0].flops_per_step is not None
                    else float("nan")
                ),
                clip_count_mean=float(np.mean([r.clip_count for r in runs])),
                threshold=thr,
            )
            out.append(cell)
        return out

    def get_cell(self, function: str, noise: str, optimizer: str) -> _Cell | None:
        for c in self.cells():
            if c.function == function and c.noise == noise and c.optimizer == optimizer:
                return c
        return None

    # ------------------------------------------------------------------ metrics
    def final_best_loss(self, function: str | None = None, noise: str | None = None) -> float:
        """Overall mean final best loss (optionally filtered)."""
        vals = []
        for c in self.cells():
            if function is not None and c.function != function:
                continue
            if noise is not None and c.noise != noise:
                continue
            vals.append(c.final_best_loss_mean)
        return float(np.mean(vals)) if vals else float("nan")

    def robustness_ratio(self, function: str, optimizer: str, noisy_noise: str) -> float:
        """``final_best(noisy) / final_best(clean)`` for a given cell pair."""
        clean = self.get_cell(function, "clean", optimizer)
        noisy = self.get_cell(function, noisy_noise, optimizer)
        if clean is None or noisy is None:
            return float("nan")
        return M.robustness_ratio(noisy.final_best_loss_mean, clean.final_best_loss_mean)

    def quantize_sweep(self, function: str, noisy_noise_prefix: str = "QuantizeNoise") -> dict[str, dict[str, float]]:
        """Return ``{optimizer: {bits_label: final_best_loss}}`` for quantization cells."""
        out: dict[str, dict[str, float]] = {}
        for c in self.cells():
            if c.function != function or c.noise == "clean":
                continue
            if c.optimizer not in out:
                out[c.optimizer] = {}
            out[c.optimizer][c.noise] = c.final_best_loss_mean
        return out

    # ------------------------------------------------------------------ export
    def df(self, long: bool = False):
        """Tidy (long-format) DataFrame if pandas is installed, else None."""
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return None
        if long:
            records = []
            for r in self._rows:
                for t in range(len(r.current_history)):
                    records.append(
                        {
                            "function": r.function,
                            "noise": r.noise,
                            "optimizer": r.optimizer,
                            "run": r.run_id,
                            "iter": t,
                            "current_loss": float(r.current_history[t]),
                            "best_loss": float(r.best_history[t]),
                        }
                    )
            return pd.DataFrame.from_records(records)
        recs = [self._cell_to_dict(c) for c in self.cells()]
        return pd.DataFrame.from_records(recs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata(),
            "config_hash": self.experiment_hash(),
            "cells": [self._cell_to_dict(c) for c in self.cells()],
        }

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with open(path, "w", newline="", encoding="ascii") as f:
            rows = [self._cell_to_dict(c) for c in self.cells()]
            if not rows:
                f.write("")
                return path
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def summary(self) -> str:
        lines = []
        header = (
            f"{'function':<10} {'noise':<18} {'opt':<10} {'succ%':>6} "
            f"{'finBest':>9} {'aucBest':>9} {'timeTot':>8}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for c in self.cells():
            lines.append(
                f"{c.function:<10} {c.noise:<18} {c.optimizer:<10} "
                f"{c.success_rate * 100:>5.0f}% {c.final_best_loss_mean:>9.4f} "
                f"{c.auc_best_loss:>9.4f} {c.wall_time_total_mean:>8.4f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ meta
    def metadata(self) -> dict[str, Any]:
        return {
            "pbit_version": __version__,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "threshold": self.threshold,
        }

    def experiment_hash(self) -> str | None:
        exp = self._experiment
        return getattr(exp, "hash", None) if exp is not None else None

    # ------------------------------------------------------------------ util
    @staticmethod
    def _cell_to_dict(c: _Cell) -> dict[str, Any]:
        return {
            "function": c.function,
            "dim": c.dim,
            "noise": c.noise,
            "optimizer": c.optimizer,
            "n_runs": c.n_runs,
            "success_rate": round(c.success_rate, 4),
            "median_hit_time_successes": c.median_hit_time_successes,
            "final_best_loss_mean": round(c.final_best_loss_mean, 6),
            "final_best_loss_median": round(c.final_best_loss_median, 6),
            "final_current_loss_mean": round(c.final_current_loss_mean, 6),
            "auc_best_loss": round(c.auc_best_loss, 4),
            "auc_current_loss": round(c.auc_current_loss, 4),
            "escape_count_current": c.escape_count_current,
            "wall_time_total_mean": round(c.wall_time_total_mean, 6),
            "wall_time_step_mean": round(c.wall_time_step_mean, 6),
            "proxy_flops_per_step": c.proxy_flops_per_step,
            "clip_count_mean": round(c.clip_count_mean, 2),
            "threshold": c.threshold,
        }
