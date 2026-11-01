"""Reproducible figures for the paper (PAPER.md).

Reads only recorded artifacts (results/*.json) and writes publication
figures to assets/figs/ as both PDF (vector, for the paper) and PNG
(150 dpi preview). No optimization is run here; every panel traces to a
checked-in results file. Deterministic: no random sampling.

Figures:
  fig1_power.pdf     power curve (C2 detectability) + null size check
  fig2_corruption.pdf  headline corruption effect: PBit vs Adam shifts
  fig3_quantize.pdf  Sphere quantization retest (or Rastrigin confound note)
  fig4_anneal.pdf    annealing diagnostics: beta(t), flip prob, entropy
  fig5_ablation.pdf  ablation: which PBit piece carries the robustness

Usage:
    python research/experiments/figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "assets" / "figs"

STYLE = {
    "pbit": "#1f77b4",
    "adam": "#d62728",
    "sgd": "#2ca02c",
    "signsgd": "#9467bd",
    "null": "#7f7f7f",
}


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "figure.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _load(name: str):
    p = ROOT / "results" / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _save(fig, stem: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{stem}.png", bbox_inches="tight", dpi=150)
    print(f"wrote assets/figs/{stem}.pdf + .png")


def fig_power(plt) -> None:
    """Detection rate vs subsample size: real effect rises, null stays flat."""
    power = _load("power.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, key, title in (
        (axes[0], "sign_noise_effect", "Sign-noise effect (C2), PBit $-$ Adam shift"),
        (axes[1], "quant4_null", "Quantization null (C1), PBit quant4 $-$ clean"),
    ):
        curve = power[key]["power_curve"] if power else []
        ns = [r["n"] for r in curve]
        det = [r["p_significant_at_0.05"] for r in curve]
        neg = [r["p_negative"] for r in curve]
        ax.plot(ns, det, "o-", color=STYLE["pbit"], label="P(significant @0.05)")
        ax.plot(ns, neg, "s--", color=STYLE["null"], label="P(negative diff)")
        ax.set_title(title)
        ax.set_xlabel("subsample size n")
        ax.set_xticks(ns)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("rate over 2000 resamples")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Power: n=5 studies cannot see the real effect; the null never fires")
    _save(fig, "fig1_power")
    plt.close(fig)


def fig_corruption(plt) -> None:
    """Headline C6 cell: paired shifts under corruption on Rastrigin."""
    claims = _load("claims.json")
    target = None
    if claims:
        for r in claims["results"]:
            if r["cid"] == "C6":
                for s in r["per_noise"]:
                    if s["landscape"] == "Rastrigin" and s["noise"] == "corruption":
                        target = s
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    if target is None:
        ax.text(0.5, 0.5, "claims.json missing: run claims.py", ha="center")
    else:
        labels = ["PBit shift", "Adam shift"]
        means = [target["pbit_shift"]["mean"], target["adam_shift"]["mean"]]
        sds = [target["pbit_shift"]["sd"], target["adam_shift"]["sd"]]
        colors = [STYLE["pbit"], STYLE["adam"]]
        bars = ax.bar(labels, means, yerr=sds, capsize=6, color=colors, alpha=0.85,
                      error_kw={"elinewidth": 1.4})
        for b, m, s in zip(bars, means, sds):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + s + 0.6,
                    f"{m:.2f} (sd {s:.2f})", ha="center", fontsize=9)
        c = target["comparison"]
        ax.set_title("Corruption on Rastrigin: noisy $-$ clean shift\n"
                     f"diff {c['mean_diff']:.1f}, CI [{c['ci_lo']:.1f}, {c['ci_hi']:.1f}], "
                     f"Holm p={c.get('p_value_holm', c['p_value']):.2g}")
        ax.set_ylabel("terminal-loss shift (noisy minus clean)")
        ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "fig2_corruption")
    plt.close(fig)


def fig_quantize(plt) -> None:
    """Sphere retest forest plot; falls back to the confound diagram."""
    retest = _load("quantize_retest.json")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    if retest is None:
        ax.text(0.5, 0.6, "quantize_retest.py has not run yet", ha="center", fontsize=11)
        ax.text(0.5, 0.4, "On Rastrigin, clip(quantized) == clip(raw): nothing to measure (C1).",
                ha="center", fontsize=9, style="italic")
        ax.set_axis_off()
        ax.set_title("Quantization retest (pending): Sphere, gradients fit the clip")
    else:
        import numpy as np

        rows = [e for e in retest["claims"]]
        ys = np.arange(len(rows))
        diffs = [e["comparison"]["mean_diff"] for e in rows]
        lo = [e["comparison"]["ci_lo"] for e in rows]
        hi = [e["comparison"]["ci_hi"] for e in rows]
        colors = [STYLE["pbit"] if e["optimizer"] == "pbit" else STYLE.get(e["optimizer"], "#555")
                  for e in rows]
        labels = [f"{e['optimizer']} {e['bits']}-bit" for e in rows]
        ax.errorbar(diffs, ys, xerr=[np.subtract(diffs, lo), np.subtract(hi, diffs)],
                    fmt="o", color="black", ecolor="black", capsize=4)
        for y, d, c in zip(ys, diffs, colors):
            ax.plot(d, y, "o", color=c, markersize=8)
        ax.set_yticks(list(ys))
        ax.set_yticklabels(labels)
        ax.axvline(0.0, color="gray", lw=1, ls="--")
        ax.set_xlabel("paired noisy-minus-clean terminal-loss shift")
        ax.set_title("Quantization on Sphere (unclipped): per-optimizer shift with 95% CI")
        ax.grid(True, axis="x", alpha=0.3)
    _save(fig, "fig3_quantize")
    plt.close(fig)


def fig_anneal(plt) -> None:
    """Annealing diagnostics: beta(t), flip prob, per-coordinate entropy."""
    import numpy as np

    from pbit.core.probability import entropy as pentropy
    from pbit.core.schedule import linear_cooling

    sched = linear_cooling(2.0, 150.0, 50.0)
    ts = np.arange(0, 600)
    betas = np.array([sched(int(t)) for t in ts])
    # A moderately-sized normalized gradient g/g_scale = 0.5.
    p = 1.0 / (1.0 + np.exp(betas * 0.5))
    ent = np.array([pentropy(np.array([v])) for v in p])
    fig, ax1 = plt.subplots(figsize=(6.4, 3.2))
    ax1.plot(ts, betas, color=STYLE["adam"], label=r"$\beta(t)$")
    ax1.set_xlabel("iteration t")
    ax1.set_ylabel(r"$\beta(t)$", color=STYLE["adam"])
    ax2 = ax1.twinx()
    ax2.plot(ts, p, color=STYLE["pbit"], label="P(+1 | g>0)")
    ax2.plot(ts, ent / np.log(2.0), color=STYLE["sgd"], ls="--", label="entropy / log 2")
    ax2.set_ylabel("probability / normalized entropy")
    ax2.set_ylim(-0.05, 1.05)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
    ax1.set_title("Annealing: exploration (high entropy) cools into descent")
    ax1.grid(True, alpha=0.3)
    _save(fig, "fig4_anneal")
    plt.close(fig)


def fig_ablation(plt) -> None:
    """Which PBit piece matters: forest plot of ablation differences."""
    abl = _load("ablation.json")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    if abl is None:
        ax.text(0.5, 0.5, "ablation.py has not run yet", ha="center", fontsize=11)
        ax.set_axis_off()
        ax.set_title("Ablation (pending): frozen-beta and deterministic-sign controls")
    else:
        import numpy as np

        rows = abl["results"]
        ys = np.arange(len(rows))
        diffs = [r["comparison"]["mean_diff"] for r in rows]
        lo = [r["comparison"]["ci_lo"] for r in rows]
        hi = [r["comparison"]["ci_hi"] for r in rows]
        labels = [r["question"] for r in rows]
        ax.errorbar(diffs, ys, xerr=[np.subtract(diffs, lo), np.subtract(hi, diffs)],
                    fmt="o", color=STYLE["pbit"], ecolor="black", capsize=4)
        ax.set_yticks(list(ys))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0.0, color="gray", lw=1, ls="--")
        ax.set_xlabel("paired difference (control minus PBit)")
        ax.set_title("Ablation on Rastrigin: no-anneal and no-noise controls vs PBit")
        ax.grid(True, axis="x", alpha=0.3)
    _save(fig, "fig5_ablation")
    plt.close(fig)


def main() -> None:
    plt = _mpl()
    fig_power(plt)
    fig_corruption(plt)
    fig_quantize(plt)
    fig_anneal(plt)
    fig_ablation(plt)
    print("all figures written to assets/figs/")


if __name__ == "__main__":
    main()



