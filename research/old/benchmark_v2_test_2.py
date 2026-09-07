"""
P-Bit Probabilistic Computing Benchmark Suite
=============================================
Compares p-bit optimizer against all standard baselines across:
  - Convergence behavior & compute cost
  - Token/iteration efficiency
  - Noise robustness (Gaussian, gradient corruption, quantization)

Usage:
    pip install numpy matplotlib scipy tqdm
    python pbit_benchmark.py

Outputs: pbit_benchmark_results/ directory with all plots + CSV
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import time
import os
import csv
from copy import deepcopy
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED        = 42
N_RUNS      = 10         # Statistical runs per optimizer
MAX_ITER    = 500
LR          = 0.01
OUTPUT_DIR  = "pbit_benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)

COLORS = {
    "PBit":              "#E85D24",
    "SGD":               "#3B8BD4",
    "Adam":              "#1D9E75",
    "RMSProp":           "#BA7517",
    "Langevin":          "#7F77DD",
    "SimAnneal":         "#D4537E",
    "EvoStrat":          "#639922",
    "Momentum":          "#0F6E56",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOSS LANDSCAPES
# ─────────────────────────────────────────────────────────────────────────────
def rastrigin(x):
    """Multi-modal: many local minima. Global min at x=0, f=0."""
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rastrigin_grad(x):
    A = 10
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

def ackley(x):
    """Deceptive landscape with many local optima."""
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + np.e

def ackley_grad(x, eps=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps; xm[i] -= eps
        g[i] = (ackley(xp) - ackley(xm)) / (2 * eps)
    return g

def rosenbrock(x):
    """Banana-shaped valley: easy to find, hard to follow."""
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

def rosenbrock_grad(x):
    g = np.zeros_like(x)
    for i in range(len(x)-1):
        g[i]   += -400*x[i]*(x[i+1]-x[i]**2) - 2*(1-x[i])
        g[i+1] +=  200*(x[i+1]-x[i]**2)
    return g

BENCHMARKS = {
    "Rastrigin": (rastrigin, rastrigin_grad, 2),
    "Ackley":    (ackley,    ackley_grad,    2),
    "Rosenbrock":(rosenbrock, rosenbrock_grad, 2),
}

# ─────────────────────────────────────────────────────────────────────────────
# NOISE INJECTORS
# ─────────────────────────────────────────────────────────────────────────────
def no_noise(g):       return g
def gaussian_noise(g, sigma=0.5):  return g + np.random.normal(0, sigma, g.shape)
def corrupt_gradient(g, p=0.2):
    mask = np.random.rand(*g.shape) < p
    gc = g.copy(); gc[mask] = -gc[mask] * 3.0
    return gc
def quantize_gradient(g, bits=4):
    levels = 2**bits - 1
    gmin, gmax = g.min(), g.max()
    if gmax == gmin: return g
    g_norm = (g - gmin) / (gmax - gmin)
    return np.round(g_norm * levels) / levels * (gmax - gmin) + gmin

NOISE_MODES = {
    "Clean":         no_noise,
    "Gaussian σ=0.5": lambda g: gaussian_noise(g, 0.5),
    "Corruption 20%": lambda g: corrupt_gradient(g, 0.2),
    "Quantize 4-bit": quantize_gradient,
}

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZERS
# ─────────────────────────────────────────────────────────────────────────────

class PBitOptimizer:
    """
    Probabilistic Bit Optimizer.
    Core idea: weight updates are stochastic binary decisions modulated
    by a temperature schedule (Boltzmann). Combines gradient direction
    with thermal exploration.

    Update rule:
        σ_i(t) ∈ {-1, +1}  with  P(σ=+1) = sigmoid(-β(t) * g_i / g_scale)
        x_i += lr * |g_i| * σ_i(t)   [magnitude-scaled step]

    Magnitude scaling ensures binary direction is decoupled from step size,
    which is gradient-proportional — fixes coarse-step problem on tight
    multimodal basins (Rastrigin basin spacing ~0.5).

    Temperature schedule: β(t) = β0 * (1 + t/τ)   [inverse temperature]
    """
    def __init__(self, lr=LR, beta0=2.0, tau=150.0):
        self.lr    = lr
        self.beta0 = beta0
        self.tau   = tau

    def step(self, x, grad, t):
        beta    = min(self.beta0 * (1 + t / self.tau), 50.0)
        g_scale = np.abs(grad).mean() + 1e-8          # normalize for β sensitivity
        prob    = 1.0 / (1.0 + np.exp(beta * grad / g_scale))  # P(σ=+1)
        sigma   = np.where(np.random.rand(*prob.shape) < prob, 1.0, -1.0)
        # Direction is binary; magnitude is gradient-proportional
        step_size = self.lr * (np.abs(grad) + 1e-8)
        return x + step_size * sigma

class SGDOptimizer:
    def __init__(self, lr=LR): self.lr = lr
    def step(self, x, grad, t): return x - self.lr * grad

class MomentumOptimizer:
    def __init__(self, lr=LR, mu=0.9):
        self.lr = lr; self.mu = mu; self.v = None
    def step(self, x, grad, t):
        if self.v is None: self.v = np.zeros_like(x)
        self.v = self.mu * self.v - self.lr * grad
        return x + self.v

class AdamOptimizer:
    def __init__(self, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = self.v = None; self.t = 0
    def step(self, x, grad, t):
        if self.m is None: self.m = np.zeros_like(x); self.v = np.zeros_like(x)
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*grad
        self.v = self.b2*self.v + (1-self.b2)*grad**2
        mh = self.m/(1-self.b1**self.t); vh = self.v/(1-self.b2**self.t)
        return x - self.lr * mh / (np.sqrt(vh) + self.eps)

class RMSPropOptimizer:
    def __init__(self, lr=LR, decay=0.9, eps=1e-8):
        self.lr = lr; self.decay = decay; self.eps = eps; self.ms = None
    def step(self, x, grad, t):
        if self.ms is None: self.ms = np.zeros_like(x)
        self.ms = self.decay*self.ms + (1-self.decay)*grad**2
        return x - self.lr * grad / (np.sqrt(self.ms) + self.eps)

class LangevinOptimizer:
    """Stochastic Gradient Langevin Dynamics: SGD + injected noise."""
    def __init__(self, lr=LR, T=0.1):
        self.lr = lr; self.T = T
    def step(self, x, grad, t):
        noise = np.sqrt(2*self.lr*self.T)*np.random.randn(*x.shape)
        return x - self.lr*grad + noise

class SimAnnealOptimizer:
    """Simulated Annealing: accept worse solutions with prob exp(-ΔE/T)."""
    def __init__(self, lr=LR, T0=5.0, cool=0.995):
        self.lr = lr; self.T = T0; self.cool = cool; self.x_best = None; self.f_best = np.inf
    def step(self, x, grad, t):
        self.T *= self.cool
        dx = -self.lr*grad + self.lr*np.random.randn(*x.shape)*self.T
        return x + dx

class EvoStratOptimizer:
    """Evolution Strategy: (μ,λ) style with population of σ-perturbed candidates."""
    def __init__(self, lr=LR, sigma=0.3, lam=10):
        self.lr = lr; self.sigma = sigma; self.lam = lam; self._fn = None; self._grad_fn = None
    def set_fns(self, fn, gfn): self._fn = fn; self._gfn = gfn
    def step(self, x, grad, t):
        perturbs = [np.random.randn(*x.shape) for _ in range(self.lam)]
        rewards = [-self._fn(x + self.sigma*p) for p in perturbs]
        rewards = np.array(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std()+1e-8)
        grad_es = -sum(r*p for r,p in zip(rewards, perturbs)) / (self.lam*self.sigma)
        return x - self.lr * grad_es

# ─────────────────────────────────────────────────────────────────────────────
# CORE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def make_optimizers(fn, gfn):
    es = EvoStratOptimizer()
    es.set_fns(fn, gfn)
    return {
        "PBit":     PBitOptimizer(),
        "SGD":      SGDOptimizer(),
        "Momentum": MomentumOptimizer(),
        "Adam":     AdamOptimizer(),
        "RMSProp":  RMSPropOptimizer(),
        "Langevin": LangevinOptimizer(),
        "SimAnneal":SimAnnealOptimizer(),
        "EvoStrat": es,
    }

def run_optimizer(opt, fn, gfn, noise_fn, dim, max_iter=MAX_ITER, clip=5.0):
    x = np.random.uniform(-2, 2, size=dim)
    history = []
    t0 = time.perf_counter()
    flops = 0
    for t in range(max_iter):
        g = gfn(x)
        flops += dim * 4  # grad evaluation (rough proxy)
        g = np.clip(noise_fn(g), -clip, clip)
        x = opt.step(x, g, t)
        flops += dim * 6  # optimizer step proxy
        history.append(float(fn(x)))
    elapsed = time.perf_counter() - t0
    return np.array(history), elapsed, flops

def run_all(fn, gfn, dim, noise_fn, n_runs=N_RUNS):
    results = {}
    for name, _ in make_optimizers(fn, gfn).items():
        runs_h, runs_t, runs_f = [], [], []
        for _ in range(n_runs):
            opts = make_optimizers(fn, gfn)  # fresh each run
            h, t, f = run_optimizer(opts[name], fn, gfn, noise_fn, dim)
            runs_h.append(h); runs_t.append(t); runs_f.append(f)
        results[name] = {
            "histories": np.array(runs_h),
            "mean":      np.mean(runs_h, axis=0),
            "std":       np.std(runs_h, axis=0),
            "time":      np.mean(runs_t),
            "flops":     np.mean(runs_f),
        }
    return results

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def iter_to_threshold(mean_hist, threshold=1.0):
    idxs = np.where(mean_hist < threshold)[0]
    return idxs[0] if len(idxs) > 0 else len(mean_hist)

def auc_loss(mean_hist):
    trapfn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return float(trapfn(mean_hist) / len(mean_hist))

def count_escapes(mean_hist, window=20, delta_threshold=0.05):
    """
    Count how many times the optimizer escapes a plateau.
    A plateau is defined as a window where loss std < delta_threshold * |mean|.
    An escape is when loss drops >5% after such a plateau.
    This is a proxy for basin-hopping ability — what reviewers actually care about.
    """
    escapes = 0
    in_plateau = False
    for i in range(window, len(mean_hist) - window):
        segment = mean_hist[i-window:i]
        local_std = np.std(segment)
        local_mean = np.abs(np.mean(segment)) + 1e-12
        if local_std / local_mean < delta_threshold:
            in_plateau = True
        elif in_plateau:
            # Check if loss dropped significantly after plateau
            before = np.mean(mean_hist[i-window:i])
            after  = np.mean(mean_hist[i:i+window])
            if before > 0 and (before - after) / before > 0.05:
                escapes += 1
            in_plateau = False
    return escapes

def final_variance(histories):
    return float(np.var(histories[:, -1]))


def compute_efficiency(results, threshold=1.0):
    rows = []
    for name, r in results.items():
        its = iter_to_threshold(r["mean"], threshold)
        auc = auc_loss(r["mean"])
        var = final_variance(r["histories"])
        eff = 1.0 / (r["time"] * max(1, its)) * 1e3
        rows.append({
            "Optimizer":         name,
            "Iter->threshold":   its,
            "AUC-Loss":          round(auc, 4),
            "Final variance":    round(var, 6),
            "Escape count":      count_escapes(r["mean"]),
            "Wall-time (s)":     round(r["time"], 4),
            "FLOPs (proxy)":     int(r["flops"]),
            "Efficiency score":  round(eff, 4),
        })
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

STYLE = dict(fontfamily="DejaVu Sans", fontsize=9)

def style_ax(ax, title, xlabel="Iteration", ylabel="Loss"):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, **STYLE)
    ax.set_ylabel(ylabel, **STYLE)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.6)

def plot_convergence(ax, results, title):
    iters = np.arange(len(next(iter(results.values()))["mean"]))
    for name, r in results.items():
        m, s = r["mean"], r["std"]
        c = COLORS[name]
        ax.semilogy(iters, np.clip(m, 1e-8, None), color=c, lw=1.6, label=name)
        ax.fill_between(iters,
                         np.clip(m-s, 1e-8, None),
                         np.clip(m+s, 1e-8, None),
                         color=c, alpha=0.10)
    style_ax(ax, title, ylabel="Loss (log)")
    ax.legend(fontsize=7, ncol=2, framealpha=0.4)

def plot_compute_bar(ax, results, metric="time", label="Wall-time (s)"):
    names  = list(results.keys())
    vals   = [results[n][metric] for n in names]
    colors = [COLORS[n] for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=0.5, width=0.6)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    style_ax(ax, f"Compute: {label}", xlabel="Optimizer", ylabel=label)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)

def plot_efficiency_radar(ax, results):
    cats  = ["Speed", "Convergence", "Stability", "Exploration"]
    N     = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize metrics to [0,1]
    opt_names = list(results.keys())
    times  = np.array([results[n]["time"] for n in opt_names])
    aucs   = np.array([auc_loss(results[n]["mean"]) for n in opt_names])
    vars_  = np.array([final_variance(results[n]["histories"]) for n in opt_names])
    # Escape count: proper basin-hopping proxy (not std variance)
    escapes = np.array([count_escapes(results[n]["mean"]) for n in opt_names], dtype=float)

    def norm_inv(a): return 1 - (a - a.min()) / ((a.max()-a.min()) + 1e-12)
    def norm_fwd(a): return (a - a.min()) / ((a.max()-a.min()) + 1e-12)
    speed  = norm_inv(times)
    conv   = norm_inv(aucs)
    stab   = norm_inv(vars_)
    expl   = norm_fwd(escapes)   # higher escapes = better exploration

    for i, name in enumerate(opt_names):
        vals = [speed[i], conv[i], stab[i], expl[i]]
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[name], lw=1.4, label=name)
        ax.fill(angles, vals, color=COLORS[name], alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, size=8)
    ax.set_yticks([]); ax.set_title("Multi-metric radar", fontsize=10, fontweight="bold", pad=16)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.4)

def plot_noise_comparison(ax, all_noise_results, opt_name, benchmark_name):
    """Final loss per noise mode for one optimizer."""
    modes = list(all_noise_results.keys())
    vals  = [all_noise_results[m][opt_name]["mean"][-1] for m in modes]
    colors = ["#3B8BD4","#E85D24","#D4537E","#BA7517"]
    bars = ax.bar(modes, vals, color=colors, edgecolor="white", linewidth=0.5, width=0.5)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    style_ax(ax, f"{opt_name} — noise sensitivity ({benchmark_name})",
             xlabel="Noise mode", ylabel="Final loss")
    ax.set_xticklabels(modes, rotation=20, ha="right", fontsize=8)

def plot_variance_heatmap(ax, results, benchmark_name):
    """Final-loss variance heatmap across optimizers (rows) and runs."""
    names = list(results.keys())
    mat   = np.array([results[n]["histories"][:, -1] for n in names])  # (opt, runs)
    im    = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Run index", fontsize=9)
    ax.set_title(f"Final-loss variance — {benchmark_name}", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

def plot_loss_landscape_2d(ax, fn, title, x_range=(-3, 3)):
    """2D heatmap of the loss landscape."""
    xs = np.linspace(*x_range, 200)
    Z  = np.array([[fn(np.array([xi, yi])) for xi in xs] for yi in xs])
    im = ax.contourf(xs, xs, Z, levels=40, cmap="plasma")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(f"Loss landscape: {title}", fontsize=10, fontweight="bold")
    ax.set_xlabel("x₁", **STYLE); ax.set_ylabel("x₂", **STYLE)
    ax.tick_params(labelsize=8)

def plot_trajectories(ax, fn, gfn, noise_fn, dim, title, x_range=(-3,3), n_steps=200):
    """2D trajectory overlay for a subset of optimizers."""
    xs = np.linspace(*x_range, 200)
    Z  = np.array([[fn(np.array([xi, yi])) for xi in xs] for yi in xs])
    ax.contourf(xs, xs, Z, levels=30, cmap="Greys", alpha=0.5)

    subset = ["PBit", "Adam", "SimAnneal", "Langevin"]
    for name in subset:
        opts = make_optimizers(fn, gfn)
        opt  = opts[name]
        x    = np.random.uniform(-2, 2, size=dim)
        traj = [x.copy()]
        for t in range(n_steps):
            g = np.clip(noise_fn(gfn(x)), -5, 5)
            x = opt.step(x, g, t)
            traj.append(x.copy())
        traj = np.array(traj)
        ax.plot(traj[:,0], traj[:,1], color=COLORS[name], lw=1.0, alpha=0.7, label=name)
        ax.scatter(traj[0,0], traj[0,1], marker="o", s=30, color=COLORS[name], zorder=5)
        ax.scatter(traj[-1,0], traj[-1,1], marker="*", s=60, color=COLORS[name], zorder=5)

    ax.set_xlim(*x_range); ax.set_ylim(*x_range)
    ax.set_title(f"Trajectories: {title}", fontsize=10, fontweight="bold")
    ax.set_xlabel("x₁", **STYLE); ax.set_ylabel("x₂", **STYLE)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, framealpha=0.4)

def plot_exploration_entropy(ax, results, title):
    """Entropy of parameter distribution as proxy for exploration."""
    for name, r in results.items():
        hists = r["histories"]   # (runs, iters)
        # Shannon entropy across runs at each step
        entropy = []
        for t in range(hists.shape[1]):
            vals = hists[:, t]
            vals_n = (vals - vals.min()) / ((vals.max()-vals.min()) + 1e-12) + 1e-12
            p = vals_n / vals_n.sum()
            entropy.append(-np.sum(p * np.log(p + 1e-12)))
        ax.plot(entropy, color=COLORS[name], lw=1.4, label=name)
    style_ax(ax, f"Exploration entropy — {title}", ylabel="H(loss distribution)")
    ax.legend(fontsize=7, ncol=2, framealpha=0.4)

def plot_flops_efficiency(ax, results):
    """Scatter: FLOPs vs AUC loss (lower-left is better)."""
    for name, r in results.items():
        auc = auc_loss(r["mean"])
        ax.scatter(r["flops"], auc, color=COLORS[name], s=100,
                   label=name, zorder=4, edgecolors="white", linewidths=0.8)
    style_ax(ax, "FLOPs vs AUC-Loss (lower-left better)",
             xlabel="FLOPs (proxy)", ylabel="AUC-Loss (avg)")
    ax.legend(fontsize=7, framealpha=0.4, ncol=2)

def plot_noise_convergence_grid(axes, all_noise_results, benchmark_name):
    """4-panel grid: convergence curve per noise mode."""
    for i, (mode, res) in enumerate(all_noise_results.items()):
        ax = axes[i]
        n_iters = len(next(iter(res.values()))["mean"])
        iters = np.arange(n_iters)
        for name, r in res.items():
            m = r["mean"]
            ax.semilogy(iters, np.clip(m, 1e-8, None), color=COLORS[name], lw=1.3, label=name)
        style_ax(ax, f"{benchmark_name} | {mode}", ylabel="Loss (log)")
        if i == 0: ax.legend(fontsize=6, ncol=2, framealpha=0.3)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN BENCHMARK RUN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  P-BIT PROBABILISTIC COMPUTING BENCHMARK SUITE")
    print("=" * 60)

    all_benchmark_results = {}
    all_noise_results_per_bench = {}

    # ── Run across all benchmarks + all noise modes ──────────────────────────
    for bname, (fn, gfn, dim) in BENCHMARKS.items():
        print(f"\n▶  Benchmark: {bname}  (dim={dim})")
        noise_results = {}
        for mname, noise_fn in NOISE_MODES.items():
            print(f"   Noise: {mname} ...", end="", flush=True)
            res = run_all(fn, gfn, dim, noise_fn)
            noise_results[mname] = res
            print(" done")
        all_benchmark_results[bname] = noise_results["Clean"]
        all_noise_results_per_bench[bname] = noise_results

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\n▶  Computing efficiency metrics...")
    all_metrics = {}
    for bname in BENCHMARKS:
        metrics = compute_efficiency(all_benchmark_results[bname])
        all_metrics[bname] = metrics

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(all_metrics[list(BENCHMARKS.keys())[0]][0].keys())
        fieldnames.insert(0, "Benchmark")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for bname, rows in all_metrics.items():
            for row in rows:
                w.writerow({"Benchmark": bname, **row})
    print(f"   Saved metrics → {csv_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 1: Convergence curves + compute bars (3 benchmarks × 2 plots)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n▶  Plotting Figure 1: Convergence + Compute...")
    fig1, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig1.suptitle("Figure 1 — Convergence & Compute Cost (Clean Gradients)",
                  fontsize=13, fontweight="bold", y=0.98)

    for row, (bname, (fn, gfn, dim)) in enumerate(BENCHMARKS.items()):
        res = all_benchmark_results[bname]
        plot_convergence(axes[row, 0], res, f"Convergence: {bname}")
        plot_compute_bar(axes[row, 1], res, "time", "Wall-time (s)")
        plot_flops_efficiency(axes[row, 2], res)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig1_convergence_compute.png")
    fig1.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"   Saved → {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 2: Noise robustness — 4×4 grid per benchmark
    # ─────────────────────────────────────────────────────────────────────────
    print("▶  Plotting Figure 2: Noise Robustness...")
    for bname in BENCHMARKS:
        fig2, axes = plt.subplots(2, 4, figsize=(20, 9))
        fig2.suptitle(f"Figure 2 — Noise Robustness: {bname}",
                      fontsize=13, fontweight="bold")

        nr = all_noise_results_per_bench[bname]

        # Row 0: convergence curves per noise mode
        plot_noise_convergence_grid(axes[0, :4], nr, bname)

        # Row 1, col 0-3: per-optimizer noise sensitivity bar charts
        for i, opt_name in enumerate(["PBit", "Adam", "SGD", "Langevin"]):
            plot_noise_comparison(axes[1, i], nr, opt_name, bname)

        plt.tight_layout()
        p = os.path.join(OUTPUT_DIR, f"fig2_noise_{bname.lower()}.png")
        fig2.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"   Saved → {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 3: Loss landscape + trajectories
    # ─────────────────────────────────────────────────────────────────────────
    print("▶  Plotting Figure 3: Loss Landscapes + Trajectories...")
    fig3, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig3.suptitle("Figure 3 — Loss Landscapes & Optimizer Trajectories",
                  fontsize=13, fontweight="bold")

    for col, (bname, (fn, gfn, dim)) in enumerate(BENCHMARKS.items()):
        plot_loss_landscape_2d(axes[0, col], fn, bname)
        np.random.seed(SEED)
        plot_trajectories(axes[1, col], fn, gfn, no_noise, dim,
                          f"Trajectories: {bname}")

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig3_landscapes_trajectories.png")
    fig3.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"   Saved → {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 4: Multi-metric radar + exploration entropy + variance heatmap
    # ─────────────────────────────────────────────────────────────────────────
    print("▶  Plotting Figure 4: Radar + Entropy + Variance...")
    bname_ref = "Rastrigin"
    res_ref   = all_benchmark_results[bname_ref]

    fig4 = plt.figure(figsize=(18, 6))
    fig4.suptitle("Figure 4 — Multi-Metric Analysis (Rastrigin, Clean)",
                  fontsize=13, fontweight="bold")

    ax_radar  = fig4.add_subplot(131, polar=True)
    ax_entropy= fig4.add_subplot(132)
    ax_var    = fig4.add_subplot(133)

    plot_efficiency_radar(ax_radar, res_ref)
    plot_exploration_entropy(ax_entropy, res_ref, bname_ref)
    plot_variance_heatmap(ax_var, res_ref, bname_ref)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig4_radar_entropy_variance.png")
    fig4.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"   Saved → {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 5: Efficiency table + AUC bar chart
    # ─────────────────────────────────────────────────────────────────────────
    print("▶  Plotting Figure 5: Efficiency Summary Table...")
    fig5, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig5.suptitle("Figure 5 — Efficiency Summary Across Benchmarks",
                  fontsize=13, fontweight="bold")

    for col, (bname, (fn, gfn, dim)) in enumerate(BENCHMARKS.items()):
        ax  = axes[col]
        res = all_benchmark_results[bname]
        rows = all_metrics[bname]
        names = [r["Optimizer"] for r in rows]
        aucs  = [r["AUC-Loss"] for r in rows]
        effs  = [r["Efficiency score"] for r in rows]
        colors= [COLORS[n] for n in names]

        x = np.arange(len(names))
        w = 0.35
        bars1 = ax.bar(x - w/2, aucs, w, label="AUC-Loss",
                       color=colors, edgecolor="white", alpha=0.85, linewidth=0.5)
        bars2 = ax.bar(x + w/2, effs, w, label="Efficiency×10³",
                       color=colors, edgecolor="white", alpha=0.45, hatch="//", linewidth=0.5)
        style_ax(ax, f"AUC-Loss vs Efficiency: {bname}",
                 xlabel="Optimizer", ylabel="Metric value")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
        ax.legend(fontsize=8, framealpha=0.4)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig5_efficiency_summary.png")
    fig5.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print(f"   Saved → {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 6: Temperature schedule of PBit + probability mass evolution
    # ─────────────────────────────────────────────────────────────────────────
    print("▶  Plotting Figure 6: PBit Temperature & Probability Analysis...")
    fig6, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig6.suptitle("Figure 6 — PBit Internal Dynamics",
                  fontsize=13, fontweight="bold")

    # Panel 1: β schedule
    ax = axes[0]
    iters = np.arange(MAX_ITER)
    beta0, tau = 1.0, 200.0
    betas = np.minimum(beta0 * (1 + iters/tau), 50.0)
    ax.plot(iters, betas, color=COLORS["PBit"], lw=2)
    style_ax(ax, "Inverse temperature β(t)", ylabel="β (inverse temp)")
    ax.fill_between(iters, betas, alpha=0.12, color=COLORS["PBit"])

    # Panel 2: P(σ=+1) for different gradient magnitudes
    ax = axes[1]
    grads = [-2, -1, -0.5, 0, 0.5, 1, 2]
    for g_val in grads:
        probs = 1/(1+np.exp(betas*g_val))
        ax.plot(iters, probs, lw=1.2,
                label=f"g={g_val:+.1f}",
                alpha=0.8)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    style_ax(ax, "P(σ=+1) vs iteration", ylabel="Probability")
    ax.legend(fontsize=7, ncol=2, framealpha=0.4)

    # Panel 3: Effective step size distribution at different β
    ax = axes[2]
    for beta_val in [0.5, 2.0, 10.0, 30.0]:
        g_range = np.linspace(-3, 3, 500)
        p_plus  = 1/(1+np.exp(beta_val*g_range))
        eff_step = p_plus - 0.5  # net bias from ±1 symmetry
        ax.plot(g_range, eff_step, lw=1.4, label=f"β={beta_val}")
    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.4)
    style_ax(ax, "Effective bias vs gradient", xlabel="Gradient value", ylabel="E[σ] bias")
    ax.legend(fontsize=7, framealpha=0.4)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig6_pbit_dynamics.png")
    fig6.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig6)
    print(f"   Saved → {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 7: Gradient quantization bit-width sweep for PBit vs Adam
    # ─────────────────────────────────────────────────────────────────────────
    print("▶  Plotting Figure 7: Quantization bit-width sweep...")
    fn, gfn, dim = list(BENCHMARKS.values())[0]  # Rastrigin
    bits_range = [1, 2, 3, 4, 6, 8, 16, 32]
    targets = ["PBit", "Adam", "SGD"]

    fig7, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig7.suptitle("Figure 7 — Quantization Robustness (Rastrigin)",
                  fontsize=13, fontweight="bold")

    final_losses = {t: [] for t in targets}
    aucs_q       = {t: [] for t in targets}

    for bits_idx, bits in enumerate(bits_range):
        qnoise = lambda g, b=bits: quantize_gradient(g, b)
        for target in targets:
            runs = []
            for run_i in range(N_RUNS):
                # Deterministic seed per (bits, target, run) — isolates quantization effect
                np.random.seed(SEED + bits_idx * 100 + run_i)
                opts = make_optimizers(fn, gfn)
                h, _, _ = run_optimizer(opts[target], fn, gfn, qnoise, dim)
                runs.append(h)
            np.random.seed(SEED)  # restore global state
            runs = np.array(runs)
            final_losses[target].append(runs[:, -1].mean())
            aucs_q[target].append(auc_loss(runs.mean(axis=0)))

    ax = axes[0]
    for t in targets:
        ax.semilogy(bits_range, final_losses[t], "o-", color=COLORS[t], lw=1.8, label=t)
    style_ax(ax, "Final loss vs quantization bits", xlabel="Bit-width", ylabel="Final loss (log)")
    ax.set_xticks(bits_range); ax.legend(fontsize=8, framealpha=0.4)

    ax = axes[1]
    for t in targets:
        ax.plot(bits_range, aucs_q[t], "s--", color=COLORS[t], lw=1.5, label=t)
    style_ax(ax, "AUC-Loss vs quantization bits", xlabel="Bit-width", ylabel="AUC-Loss")
    ax.set_xticks(bits_range); ax.legend(fontsize=8, framealpha=0.4)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig7_quantization_sweep.png")
    fig7.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig7)
    print(f"   Saved → {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 8: Escape count + noise degradation ratio
    # ─────────────────────────────────────────────────────────────────────────
    print("▶  Plotting Figure 8: Escape Count + Noise Degradation Ratio...")
    fig8, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig8.suptitle("Figure 8 — Basin Escape Ability & Noise Degradation Ratio",
                  fontsize=13, fontweight="bold")

    # Panel 1: Escape counts per optimizer per benchmark
    ax = axes[0]
    bnames   = list(BENCHMARKS.keys())
    opt_list = list(all_benchmark_results[bnames[0]].keys())
    x_pos    = np.arange(len(opt_list))
    bar_w    = 0.25
    b_colors = ["#3B8BD4", "#1D9E75", "#BA7517"]
    for bi, bn in enumerate(bnames):
        escapes = [count_escapes(all_benchmark_results[bn][o]["mean"]) for o in opt_list]
        bars = ax.bar(x_pos + bi*bar_w, escapes, bar_w,
                      label=bn, color=b_colors[bi], edgecolor="white",
                      linewidth=0.5, alpha=0.85)
        ax.bar_label(bars, fmt="%d", fontsize=6, padding=1)
    style_ax(ax, "Basin escape count per optimizer", xlabel="Optimizer", ylabel="# Escapes")
    ax.set_xticks(x_pos + bar_w); ax.set_xticklabels(opt_list, rotation=30, ha="right", fontsize=7)
    ax.legend(fontsize=8, framealpha=0.4)

    # Panel 2: Noise degradation ratio (final_loss_noisy / final_loss_clean) — lower is better
    ax = axes[1]
    bname_nd = "Rosenbrock"   # most differentiated in the data
    nr_rosen = all_noise_results_per_bench[bname_nd]
    noise_modes_plot = ["Gaussian σ=0.5", "Corruption 20%", "Quantize 4-bit"]
    nm_colors = ["#E85D24", "#D4537E", "#BA7517"]
    opt_nd = list(nr_rosen["Clean"].keys())
    x_nd = np.arange(len(opt_nd))
    bw_nd = 0.25
    for ni, nm in enumerate(noise_modes_plot):
        ratios = []
        for o in opt_nd:
            clean = nr_rosen["Clean"][o]["mean"][-1] + 1e-8
            noisy = nr_rosen[nm][o]["mean"][-1] + 1e-8
            ratios.append(noisy / clean)
        bars = ax.bar(x_nd + ni*bw_nd, ratios, bw_nd,
                      label=nm, color=nm_colors[ni], edgecolor="white",
                      linewidth=0.5, alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6, label="No degradation")
    style_ax(ax, f"Noise degradation ratio ({bname_nd})",
             xlabel="Optimizer", ylabel="Noisy loss / Clean loss (lower=better)")
    ax.set_xticks(x_nd + bw_nd); ax.set_xticklabels(opt_nd, rotation=30, ha="right", fontsize=7)
    ax.legend(fontsize=7, framealpha=0.4)

    # Panel 3: PBit vs Adam — convergence stability (std band width over iterations)
    ax = axes[2]
    compare_opts = ["PBit", "Adam", "SGD", "SimAnneal"]
    iters_plot = np.arange(len(all_benchmark_results["Rastrigin"]["PBit"]["mean"]))
    for o in compare_opts:
        r   = all_benchmark_results["Rastrigin"][o]
        std = r["std"]
        mn  = r["mean"]
        # Coefficient of variation = std/mean: scale-free stability measure
        cv  = std / (np.abs(mn) + 1e-8)
        ax.plot(iters_plot, cv, color=COLORS[o], lw=1.4, label=o)
    style_ax(ax, "Convergence stability (CV = std/mean, Rastrigin)",
             ylabel="Coefficient of variation (lower=more stable)")
    ax.legend(fontsize=8, framealpha=0.4)
    ax.set_ylim(0, 3)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig8_escape_noise_stability.png")
    fig8.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig8)
    print(f"   Saved → {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # PRINT SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  EFFICIENCY SUMMARY TABLE")
    print("="*80)
    header = f"{'Benchmark':<12} {'Optimizer':<12} {'Iter->thresh':>12} {'AUC-Loss':>10} {'Variance':>10} {'Time(s)':>9} {'Eff.score':>10}"
    print(header)
    print("-"*80)
    for bname, rows in all_metrics.items():
        for r in rows:
            print(f"{bname:<12} {r['Optimizer']:<12} {r['Iter->threshold']:>12} "
                  f"{r['AUC-Loss']:>10.4f} {r['Final variance']:>10.6f} "
                  f"{r['Wall-time (s)']:>9.4f} {r['Efficiency score']:>10.4f}")

    print("\n" + "="*60)
    print("  ALL DONE")
    print(f"  Figures saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("  Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    {f}")
    print("="*60)


if __name__ == "__main__":
    main()