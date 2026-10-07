"""Experiment runner: reproducible multi-optimizer / multi-noise benchmark.

Pipeline per optimizer step (for gradient optimizers):

    x_t
    raw_grad = fn.grad(x_t)
    noisy_grad = noise(raw_grad, rng_noise)   # shared noise stream (comparable)
    clipped_grad = clip(noisy_grad)
    x_{t+1} = opt.step(x_t, clipped_grad, t)
    current_loss = fn(x_{t+1})
    best_loss = min(best_loss, current_loss)

Randomness model (see `pbit.core.rng`): three SHA256-derived streams per cell,
with ``rng_init`` and ``rng_noise`` shared across optimizers (common random
numbers -> paired starts / comparable noisy gradients) and ``rng_opt`` private
per optimizer.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from pbit.bench.report import Report, _RunRow
from pbit.bench.specs import FunctionSpec, NoiseSpec, OptimizerSpec, config_hash
from pbit.core.rng import derive_rng
from pbit.optim.base import AskTellOptimizer, GradientOptimizer

DEFAULT_CLIP = 5.0

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def _tqdm(iterable, **kwargs):
        return iterable


@dataclass
class Experiment:
    """Encapsulates and runs a benchmark experiment.

    Parameters
    ----------
    config:
        A dict (or object with attributes) containing:
          - ``functions``: list[FunctionSpec]
          - ``noises``: list[NoiseSpec | None]  (None -> clean)
          - ``optimizers``: list[OptimizerSpec]
          - ``max_iter``: int
          - ``n_runs``: int
          - ``seed``: int
          - ``clip``: float
    """

    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        c = self.config
        self.functions: Sequence[FunctionSpec] = c.get("functions", [])
        self.noises: Sequence[NoiseSpec | None] = c.get("noises", [None])
        self.optimizers: Sequence[OptimizerSpec] = c.get("optimizers", [])
        self.max_iter: int = int(c.get("max_iter", 500))
        self.n_runs: int = int(c.get("n_runs", 10))
        self.seed: int = int(c.get("seed", 42))
        self.clip: float = float(c.get("clip", DEFAULT_CLIP))

        if not self.functions:
            raise ValueError("Experiment requires at least one function")
        if not self.optimizers:
            raise ValueError("Experiment requires at least one optimizer")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter!r}")
        if self.n_runs <= 0:
            raise ValueError(f"n_runs must be positive, got {self.n_runs!r}")

    @property
    def hash(self) -> str:
        return config_hash(
            list(self.functions),
            list(self.noises),
            list(self.optimizers),
            seed=self.seed,
            max_iter=self.max_iter,
            n_runs=self.n_runs,
            clip=self.clip,
        )

    def run(self) -> Report:
        report = Report(experiment=self)
        rows: list[_RunRow] = []

        total = len(self.functions) * len(self.noises) * len(self.optimizers) * self.n_runs
        pbar = _tqdm(total=total, desc="Benchmarking", unit="run")

        for fnspec in self.functions:
            for nspec in self.noises:
                for ospec in self.optimizers:
                    for run in range(self.n_runs):
                        # Stream design (common random numbers):
                        # - init: keyed on (function, run) only, so every noise
                        #   condition and every optimizer shares one start point.
                        # - noise: keyed on (function, noise, run), so optimizers
                        #   within one noise condition see the same perturbation.
                        # - opt: keyed on (function, run, optimizer), so a
                        #   stochastic optimizer's private coins are identical
                        #   across noise conditions but independent across
                        #   optimizers.
                        # Do not add the noise or optimizer name to a stream that
                        # is meant to be shared: that silently unpairs results.
                        rng_init = derive_rng(
                            self.seed, f"f={fnspec.name}", f"run={run}", "init"
                        )
                        rng_noise = derive_rng(
                            self.seed,
                            f"f={fnspec.name}",
                            f"n={nspec.name if nspec else 'clean'}",
                            f"run={run}",
                            "noise",
                        )
                        rng_opt = derive_rng(
                            self.seed,
                            f"f={fnspec.name}",
                            f"run={run}",
                            f"o={ospec.name}",
                            "opt",
                        )
                        row = self._run_one(
                            fnspec, nspec, ospec, run, rng_init, rng_noise, rng_opt
                        )
                        rows.append(row)
                        pbar.update(1)

        pbar.close()
        report.set_rows(rows)
        return report

    # ------------------------------------------------------------- internal
    def _run_one(
        self,
        fnspec: FunctionSpec,
        nspec: NoiseSpec | None,
        ospec: OptimizerSpec,
        run_id: int,
        rng_init: np.random.Generator,
        rng_noise: np.random.Generator,
        rng_opt: np.random.Generator,
    ) -> _RunRow:
        # Initialize inside shared init stream (common random numbers).
        lo, hi = fnspec.init_bounds
        x = rng_init.uniform(lo, hi, size=fnspec.dim)

        opt = ospec.factory()

        dim = fnspec.dim
        current_history = np.empty(self.max_iter)
        best_history = np.empty(self.max_iter)
        step_times = np.empty(self.max_iter)
        clip_count = 0
        n_evals = 0
        diagnostics: dict | None = None

        best = float("inf")

        start = time.perf_counter()
        if isinstance(opt, GradientOptimizer) or hasattr(opt, "step"):
            opt.reset(rng_opt)
            grad_fn = fnspec.grad if fnspec.grad is not None else lambda x: _fd_grad(fnspec, x)
            for t in range(self.max_iter):
                raw = np.asarray(grad_fn(x), dtype=np.float64)
                noisy = nspec.apply(raw, rng_noise) if nspec is not None else raw
                clipped = _clip_grad(noisy, self.clip)
                if np.any(np.abs(noisy) > self.clip):
                    clip_count += 1
                t0 = time.perf_counter()
                x = opt.step(x, clipped, t)
                step_times[t] = time.perf_counter() - t0
                cur = float(fnspec.evaluate(x))
                n_evals += 1
                best = min(best, cur)
                current_history[t] = cur
                best_history[t] = best
            diagnostics = opt.state_dict() if hasattr(opt, "state_dict") else None
        elif isinstance(opt, AskTellOptimizer):
            opt.reset(rng_opt)
            for t in range(self.max_iter):
                cand = np.asarray(opt.ask(), dtype=np.float64)
                # Exactly one objective evaluation per iteration, matching the
                # gradient path. Evaluating here and again in tell() doubled the
                # ask/tell budget and invalidated every reported comparison.
                cur = float(fnspec.evaluate(cand))
                n_evals += 1
                best = min(best, cur)
                t0 = time.perf_counter()
                opt.tell(cur)
                step_times[t] = time.perf_counter() - t0
                current_history[t] = cur
                best_history[t] = best
        else:  # pragma: no cover
            raise TypeError(f"unsupported optimizer type: {type(opt)}")

        total = time.perf_counter() - start

        flops = ospec.flops_per_step(dim) if ospec.flops_per_step is not None else None

        return _RunRow(
            function=fnspec.name,
            dim=fnspec.dim,
            noise=nspec.name if nspec is not None else "clean",
            optimizer=ospec.name,
            run_id=run_id,
            current_history=current_history,
            best_history=best_history,
            step_times=step_times,
            wall_time_total=total,
            clip_count=clip_count,
            diagnostics=diagnostics or {},
            flops_per_step=flops,
            n_evals=n_evals,
        )


def _clip_grad(g: np.ndarray, clip: float) -> np.ndarray:
    return np.clip(g, -clip, clip)


def _fd_grad(fnspec: FunctionSpec, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Vectorized central finite-difference gradient fallback."""
    x = np.asarray(x, dtype=np.float64)
    dim = len(x)
    xp = np.tile(x, (dim, 1))
    xm = np.tile(x, (dim, 1))
    for i in range(dim):
        xp[i, i] += eps
        xm[i, i] -= eps
    fp = np.apply_along_axis(fnspec.evaluate, 1, xp)
    fm = np.apply_along_axis(fnspec.evaluate, 1, xm)
    return (fp - fm) / (2 * eps)
