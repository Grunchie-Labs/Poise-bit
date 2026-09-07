"""Optional matplotlib visualizations (imported lazily; not required to run).

Production of the "blog-ready" figures:

- ``plot_convergence`` : best-so-far convergence curves for a function.
- ``plot_quantization_sweep`` : final best loss vs bit-width.
- ``plot_pbit_expectation`` : E[dx] vs normalized gradient for several beta.
- ``plot_flip_probability`` : heatmap of P(sigma=+1) vs (gradient, beta).
- ``plot_diagnostics`` : beta(t), entropy(t), mean flip probability(t).

These only import matplotlib at call time so the benchmark core stays headless.
"""

from __future__ import annotations

import numpy as np

from pbit.bench.report import Report


def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_convergence(report: Report, function: str, noise: str = "clean", ax=None, colors=None):
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for c in report.cells():
        if c.function != function or c.noise != noise:
            continue
        # We recompute the per-run mean best curve by re-aggregating rows.
        bests = _mean_best_curve(report, function, noise, c.optimizer)
        color = (colors or {}).get(c.optimizer)
        ax.semilogy(np.clip(bests, 1e-8, None), lw=1.6, label=c.optimizer, color=color)
    ax.set_title(f"Best-so-far: {function} ({noise})")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best loss (log)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def _mean_best_curve(report: Report, function: str, noise: str, optimizer: str) -> np.ndarray:
    rows: list[_RunRow] = [
        r
        for r in report.rows
        if r.function == function and r.noise == noise and r.optimizer == optimizer
    ]
    if not rows:
        return np.array([])
    return np.mean(np.stack([r.best_history for r in rows]), axis=0)


def plot_quantization_sweep(report: Report, function: str, ax=None):
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sweep = report.quantize_sweep(function)
    bits_labels = []
    series: dict[str, list[float]] = {}
    for noise_label in sorted(
        {c.noise for c in report.cells() if c.function == function and c.noise != "clean"}
    ):
        _bits = _extract_bits(noise_label)
        if _bits is None:
            continue
        bits_labels.append(_bits)
        for opt, vals in sweep.items():
            if noise_label in vals:
                series.setdefault(opt, []).append(vals[noise_label])
    order = np.argsort(bits_labels)
    bits_sorted = [bits_labels[i] for i in order]
    for opt, ys in series.items():
        ys_sorted = [ys[i] for i in order]
        ax.semilogy(bits_sorted, np.clip(ys_sorted, 1e-8, None), "o-", lw=1.6, label=opt)
    ax.set_xticks(bits_sorted)
    ax.set_xlabel("bit-width (stochastic quantize)")
    ax.set_ylabel("final best loss (log)")
    ax.set_title(f"Quantization sweep: {function}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def _extract_bits(noise_label: str) -> int | None:
    try:
        return int(noise_label.split("=")[-1])
    except (ValueError, IndexError):
        return None


def plot_pbit_expectation(betavals=(0.5, 2.0, 10.0, 30.0), ax=None):
    """E[sigma] bias vs normalized gradient for several inverse temperatures."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    from pbit.core.probability import sigmoid

    g = np.linspace(-3, 3, 500)
    for beta in betavals:
        p = sigmoid(-beta * g)
        eff = p - 0.5
        ax.plot(g, eff, lw=1.4, label=f"β={beta}")
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("normalized gradient")
    ax.set_ylabel("E[σ] bias")
    ax.set_title("PBit update expectation vs gradient")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_flip_probability(betavals=(0.5, 2.0, 10.0, 30.0), ax=None):
    """P(sigma=+1) vs normalized gradient for several inverse temperatures."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    from pbit.core.probability import sigmoid

    g = np.linspace(-3, 3, 500)
    for beta in betavals:
        p = sigmoid(-beta * g)
        ax.plot(g, p, lw=1.4, label=f"β={beta}")
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("normalized gradient")
    ax.set_ylabel("P(σ=+1)")
    ax.set_title("Boltzmann flip probability")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_diagnostics(ax=None):
    """beta(t), entropy proxy, and mean flip-probability proxy over annealing."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    from pbit.core.schedule import linear_cooling

    tau = 150.0
    schedule = linear_cooling(2.0, tau, 50.0)
    ts = np.arange(0, 800)
    betas = np.array([schedule(t) for t in ts])
    # flip prob for a moderately-sized normalized gradient (e.g. g/g_scale = 0.5)
    p = 1.0 / (1.0 + np.exp(betas * 0.5))
    ax.plot(ts, betas / betas.max(), label="β(t)/βmax")
    ax.plot(ts, p, label="P(σ=+1) g=0.5")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.set_title("Annealing diagnostics")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax
