# PBit → Production-Grade PyPI Library — Implementation Plan

**Project:** PBit (probabilistic/computational p-bit computing library)
**Target:** Production-quality PyPI package + strong research/engineering flagship for **Grunchie Labs**
**Phase:** Audit & Planning (no repository files modified except this document)

---

# 1. Executive Summary

The current repository is an **early-stage research prototype**, not a library. It contains **two
independent, near-duplicate benchmark scripts** (`test_2.py`, `Untitled-1.py`), a **toy PyTorch
transformer** (`pbit_attention.py` + a Jupyter notebook of the same code), and a folder of
**benchmark output figures** (`pbit_benchmark_results/`). There is no package structure, no
installer metadata, no tests, no docs, and no version control.

The single most valuable asset is a **well-engineered PBit optimizer update rule** — stochastic
binary (sigma +1/−1) weight updates driven by a Boltzmann sigmoid modulated by an inverse-temperature
annealing schedule, with a magnitude-scaled step that decouples binary *direction* from
gradient-proportional *step size*. Around it sits a genuinely useful but unscalable benchmark harness
with meaningful metrics (AUC-loss, iteration-to-threshold, escape count, noise-robustness).

**Strategy:** Rebuild as a small, focused package. Ship two flagship pillars:

1. **`pbit.optim` — the p-bit stochastic optimizer** (the research/engineering core).
2. **`pbit.bench` — a reproducible, hardware-vs-proxy-aware benchmarking harness** (differentiating
   infrastructure).

The transformer attention experiment is a **separate research thread** — include only if it evolves
into a genuine stochastic-attention module; otherwise keep it out of the core library scope.

Repository roll-up in three lines:

- `test_2.py` and `Untitled-1.py` → become `pbit/optim/*` + `pbit/bench/*` (one copy, rewritten).
- `pbit_attention.py` / notebook → becomes a *research notes + example* thread, not core.
- `pbit_benchmark_results/` → regenerated output; never committed as source.

---

# 2. Current Repository Analysis

## 2.1 Folder Structure

```
p_bit_net/
├── pbit_attention.py                  # PyTorch toy transformer (experimental)
├── Interactive - pbit_attention.py.ipynb  # Notebook copy of pbit_attention.py
├── test_2.py                          # PBit benchmark suite v2 (refined, 844 lines)
├── Untitled-1.py                      # PBit benchmark suite v1 (735 lines, earlier)
└── pbit_benchmark_results/
    ├── metrics_summary.csv            # 24-row summary of 8 optimizers × 3 benchmarks
    └── fig1..fig8_*.png               # 8 multi-panel matplotlib figures
```

## 2.2 Files & Roles

| File | Role | Status |
|------|------|--------|
| `test_2.py` | Primary benchmark/experiment script. Contains PBit optimizer + 7 baselines, 3 benchmark functions, 4 noise modes, 5 metrics, `main()` orchestration producing 8 figures + CSV. | KEEP (refactor into library) |
| `Untitled-1.py` | Earlier version of `test_2.py`. Nearly identical, **no `count_escapes` metric**, no `EvoStrat.set_fns` cleanup issues, has old `plot_efficiency_radar` (uses mean-std instead of escape count). Represents v1; v2 supersedes it. | REMOVE (duplicate) |
| `pbit_attention.py` | Toy Transformer in PyTorch: positional encoding, multihead attention with **temperature-scaled softmax**, entropy/sparsity metrics, AdamW training on synthetic next-token data. Has a **syntax bug** `p_bit_weight=` on line 117. | REDESIGN / separate thread |
| `Interactive - pbit_attention.py.ipynb` | Jupyter copy of the same attention code (no `p_bit_weight` bug), plus an "experiments to try" docstring (stochastic scores, p-bit edge sampling, dynamic temperature, ising couplings, annealing, entropy tracking). | REDESIGN / separate thread |
| `pbit_benchmark_results/*` | Generated figures + CSV from running `test_2.py`. | REMOVE from source (regenerate) |

## 2.3 Core Algorithms

### The PBit optimizer (the crown jewel) — `test_2.py:121-151`
```
β(t)  = min(β0 · (1 + t/τ), β_max)            # inverse temperature, anneals up (cools)
g_scale = mean(|g|) + ε                       # per-step normalization
P(σ=+1) = sigmoid(-β(t) · g / g_scale)        # Boltzmann probability of +1
σ       = Bernoulli(P) ∈ {−1, +1}             # binary stochastic decision
step    = lr · (|g| + ε) · σ                  # magnitude-scaled step (direction binary, size ∝ |g|)
x      += step
```
**Key insight (v2 only, `test_2.py:131-136`):** earlier version (`Untitled-1.py:139-144`) used
`x += lr·σ` — a *fixed* magnitude step independent of gradient size. PBit-2 changed this to
`lr·|g|·σ`, decoupling the **binary direction** decision (thermal) from the **gradient-proportional
step size** (magnitude). This is the differentiation that actually makes PBit competitive on
multimodal basins (e.g. Rosenbrock converges at iteration 51 vs 500 for many baselines). **This v2
rule is the algorithm to preserve exactly.**

### Baselines (all `test_2.py:152-213`)
SGD, Momentum, Adam, RMSProp, Langevin (SGLD-style), Simulated Annealing, Evolution Strategy (μ/λ).
These are **standard references used for comparison**, not research objects. Many can be swapped to
well-tested libraries, but self-contained NumPy versions are fine for reproducibility.

### Benchmark functions (`test_2.py:52-92`)
- **Rastrigin** (multi-modal, many local minima) + analytic gradient.
- **Ackley** (deceptive local optima) + **finite-difference** gradient (coordinate-wise, Python loop).
- **Rosenbrock** (banana valley) + analytic gradient.
All defined only for **dim=2** in the benchmark registry (functions are general-dimensional).

### Noise injectors (`test_2.py:97-115`)
- `no_noise` (clean)
- `gaussian_noise(g, σ=0.5)`
- `corrupt_gradient(g, p=0.2)` (flip 20% of components ×(−3))
- `quantize_gradient(g, bits=4)` (uniform quantization)
Used to measure **noise robustness** — a genuinely interesting axis (PBit is designed to be robust,
since it is inherently stochastic).

### Metrics (`test_2.py:268-322`)
- `iter_to_threshold` — first iteration below loss threshold (default 1.0); 500 = never reached.
- `auc_loss` — normalized area under loss curve (lower better).
- `final_variance` — variance of final loss across runs.
- `count_escapes` (**v2 only**) — counts basin-hopping escapes (plateau → >5% drop). Designed as a
  reviewer-facing exploration proxy. Forward-looking and clever.
- `efficiency_score` = 1000 / (wall-time × max(1, iters-to-threshold)).
- `FLOPs (proxy)` = rough `dim·4` + `dim·6` per iteration (NOT measured, estimated).

## 2.4 How Components Interact

```
BENCHMARKS → f(x), ∇f(x)  ─┐
NOISE_MODES → g̃ = noise(∇f) ┴→ run_optimizer(opt, fn, gfn, noise_fn, dim)
                                       │  x ~ U(-2,2); for t: g=∇f(x); g=clip(noise(g)); x=opt.step(x,g,t)
                                       ▼
                              histories, wall-time, flops
                                       │  × N_RUNS → mean/std
                                       ▼
                            run_all → results{optimizer: {histories, mean, std, time, flops}}
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
             compute_efficiency (metrics)          plot_* helpers (matplotlib)
                    │                                     │
                    ▼                                     ▼
            metrics_summary.csv                    fig1..fig8 PNG files
```

Everything is **top-level imperative code** with module-level constants (SEED=42, N_RUNS, MAX_ITER,
LR, OUTPUT_DIR). There is **no class/API boundary** separating optimizers from metrics from plotting.

## 2.5 Dependencies
- **NumPy** (core math, RNG via `np.random`)
- **Matplotlib** (plotting)
- **tqdm** (imported in both scripts but effectively unused in the orchestrator)
- **Torch** (only `pbit_attention.py`; NOT used by benchmarks)
- **scipy** — mentioned in a usage comment but never imported.
- Python stdlib: `time, os, csv, copy, math, dataclasses`.

## 2.6 Tests / Docs / Examples / Version Control
- **Tests:** none.
- **Docs:** none (only docstring comments).
- **Examples:** none beyond the scripts themselves.
- **Git:** not a repository (`.git` absent).
- **Config/changelog/pyproject:** none.

## 2.7 Technical Debt & Problems
1. **Code duplication:** `test_2.py` and `Untitled-1.py` are ~93% identical. Keeping both invites
   divergent bugs (they already differ: PBit magnitude scaling, escape metric, radar metric).
2. **No separation of concerns:** optimizers, benchmark functions, noise, metrics, RNG, and
   matplotlib plotting all live in one flat file with module-level global state.
3. **RNG hygiene is broken (reproducibility-critical):** Global `np.random.seed(SEED)` at module top
   plus in-loop `np.random.seed(SEED + ...)` calls re-seed and then **restore** global state
   (`test_2.py:724`). This is fragile: run-ordering, iteration ordering, and the shared global stream
   make exact reproducibility machine-fragile. `EvoStrat` samples its own random draws from the same
   global stream in a way that shifts PBit's stream depending on which optimizer is on which
   iterations.
4. **FLOPs proxy is fabricated, not measured** (`flops += dim*4`, `dim*6`), so it is identical
   (10000) across every optimizer in the CSV — it carries **zero discriminating information** as
   implemented and is conflated with real wall-time in the "efficiency" score.
5. **Efficiency score mixes measured time with proxy/variance** — compares `1/(time·iters)` across
   optimizers where PBit is artificially penalized (more `np.exp`/`np.where` work per step) — a
   measure of *implementation overhead*, not algorithmic merit.
6. **Global-constant tuning:** LR=0.01, β0=2.0, τ=150 are baked in; no hyperparameter plumbing.
   Adam uses default LR across all problems, so conclusions are tuned-to-PBit by construction.
7. **`Ackley` gradient is a Python-loop finite difference** — slow and uses perturbation eps, while
   PBit's `g_scale` uses `mean(|g|)`; inconsistent gradient quality across benchmarks.
8. **Dead code / unused imports:** `gridspec`, `FancyArrowPatch`, `deepcopy`, `tqdm` imported but
   unused. `SimAnnealOptimizer` stores `x_best/f_best` never used.
9. **Syntax bug** in `pbit_attention.py:117` (`p_bit_weight=`) → the file will not run as-is.
10. **Headless/plot coupling:** producing figures requires matplotlib backend; the benchmark should
    run (and export CSV/JSON) without any plotting dependency.

## 2.8 Current Public "API" (informal)
- `PBitOptimizer(lr, beta0, tau).step(x, grad, t) -> x'`
- Other optimizers: same `.step(x, grad, t) -> x'` signature (EvoStrat additionally needs `.set_fns`).
- `run_optimizer`, `run_all`, `compute_efficiency`, metrics, plot helpers.
- Benchmark functions: `rastrigin(x)`, `ackley(x)`, `rosenbrock(x)` (+ `*_grad`).

---

# 3. Feature Inventory

Legend: KEEP / REFACTOR / REDESIGN / REMOVE / NEW FEATURE

## 3.1 Core PBit functionality

| Feature | Location | Class. | Rationale |
|---------|----------|--------|-----------|
| PBit update rule (v2, magnitude-scaled) | `test_2.py:121` | **KEEP** (core) | Finest asset; exact preserving of `lr·|g|·σ` logic |
| PBit previous rule (v1, fixed-step) | `Untitled-1.py:139` | REFACTOR → become a separate `step_size` strategy, not a duplicate class | Archive as a variant, expose via strategy param |
| Inverse-temperature anneal schedule `β(0)=β0(1+t/τ)`, capped | both | **KEEP** | Core schedule; make pluggable |
| APositive/negative sigma Bernoulli sampling | both | **KEEP** (extract) | The "p-bit" sampling primitive |
| Probability of transition `P(σ=+1)=sigmoid(-βg)` | both | **KEEP** (extract) | Boltzmann transition; expose as `PBitSampler` |
| Gradient magnitude normalization `g_scale` | `test_2.py:145` | **KEEP** | Critical for β sensitivity |
| P-bit attention (thermo softmax / entropy metrics) | `pbit_attention.py` | **REDESIGN** | Research thread; not deterministic, no tests, buggy .py |

## 3.2 Optimization

| Feature | Class. | Rationale |
|---------|--------|-----------|
| `PBitOptimizer` (step + schedule) | **KEEP** → `pbit/optim/pbit.py` | Flagship |
| `SGD/Momentum/Adam/RMSProp` | KEEP (REFACTOR to uniform `Optimizer` base) | Needed as reference baselines; reuse known-good math |
| `Langevin` (SGLD) | KEEP (REFACTOR) | Probabilistic baseline; informative contrast to PBit |
| `SimAnneal` | KEEP (REFACTOR) | Standard stochastic baseline |
| `EvoStrat (μ,λ)` | **REMOVE** (as a 1st-class optimizer) or REFACTOR into `pbit.bench.baselines` | Needs `set_fns` (broken uniformity); slowest; keep only as baseline |
| Gradient-based `step(x, grad, t)` interface | **REDESIGN** into clean `Optimizer` protocol | Currently not a real protocol (EvoStrat breaks it) |
| Temperature schedule abstraction | KEEP → **NEW** `schedule.py` | Enable linear/plateau/adaptive schedules |

## 3.3 Benchmarking

| Feature | Class. | Rationale |
|---------|--------|-----------|
| Benchmark registry (Rastrigin/Ackley/Rosenbrock) | **KEEP** → new `pbit/bench/functions.py` | Solid, well-known, dim-general functions |
| Analytic + finite-diff gradients | **KEEP** but provide true gradients for all | Replace slow loop FD gradient for Ackley with analytic/vectorized |
| Noise injectors (gaussian/corrupt/quantize) | **KEEP** → `pbit/bench/noise.py` | Differentiating robustness axis |
| Metrics (iter-to-threshold, AUC, variance, escapes) | **KEEP** → `pbit/bench/metrics.py` | Redesign `escape` definition to be less parameter-sensitive; retain |
| Efficiency score / FLOPs proxy | **REDESIGN** | Current proxy is fabricated; split measured-time vs proxy; phase-1 drop the fake FLOPs |
| `run_optimizer`/`run_all` runner | **REDESIGN** → `BenchmarkRunner` | Separate RNG, measure time cleanly, per-optimizer seeding |
| Plot helpers (8 figures) | **KEEP** → optional `pbit/bench/plot.py` (import-guarded) | Move behind matplotlib-optional check so core runs headless |
| Comparison infrastructure (multi-optimizer, multi-noise, multi-benchmark) | **KEEP** → `Experiment`/`Report` | Now orchestrated in `main()`; make reusable |

## 3.4 Infrastructure

| Feature | Class. | Rationale |
|---------|--------|-----------|
| Global RNG / seeding | **REDESIGN** | Per-run seeded `np.random.Generator`; no module-global reseeding in loop |
| Module constants (SEED, N_RUNS, MAX_ITER) | **KEEP** → config dataclass | Now explicit config |
| Serialization | **NEW** (CSV + JSON/arrow of results + metadata) | Currently only a hand-rolled CSV with non-ASCII headers (`→`) that break portability |
| Logging | **NEW** (`logging` + progress) | Replace `print()` |
| Performance utility (time measurement) | KEEP/REFACTOR | Use `time.perf_counter`, report distribution not just mean |
| Backend abstraction | **NEW** (minimal) | NumPy first; design seam for future Rust/Triton but do NOT build now |
| Bounds/clipping, float stability | **KEEP** (clip=5.0, +1e-8) | Preserve; make configurable |

---

# 4. Grunchie Labs Flagship Features

Ranked. These should become the package's identity and the marketing/research story.

## #1 — The PBit Stochastic Optimizer (`pbit.optim.PBitOptimizer`)
1. **Current behavior:** binary-direction stochastic gradient descent driven by a Boltzmann
   probability `P(σ=+1)=sigmoid(-β·g/g_scale)` with inverse-temperature annealing and
   gradient-proportional magnitude scaling.
2. **Why valuable:** It's the true differentiator. It sits at a credible research intersection —
   probabilistic computing / p-bits (Memristor/p-bit *logic & computing* literature), Langevin
   thermodynamics, and optimization. It demonstrably escapes local minima on Rosenbrock (converges
   at iter 51 vs 500 for SGD/Adam) and is robust to gradient quantization/corruption by construction.
   This is a publishable, physically-motivated optimizer, not another Adam variant.
3. **What must be rewritten:** wrap as a class with a *strategy* pattern for (a) transition
   probability, (b) temperature schedule, (c) step-size rule; thread a caller-supplied RNG; add
   per-parameter and momentum/flavors later. Extract the tiny probability/anneal helpers into
   testable modules.
4. **Public API:**
   ```python
   from pbit.optim import PBitOptimizer, linear_cooling

   opt = PBitOptimizer(lr=0.01, beta0=2.0, tau=150.0,
                       schedule=linear_cooling, seed=42)
   x = opt.step(x, grad, t)          # returns new params
   opt.beta(t)                        # current inverse temperature (inspectable)
   ```
   and lower-level primitives:
   ```python
   from pbit.core import sigmoid, bernoulli_bit, sigmoid_probability, normalize_scale
   ```
5. **Technically interesting:** the connection between a *single stochastic binary unit* (the
   p-bit), Boltzmann statistics, and optimization; a clean thermal annealing phase transition
   between exploration (β→0) and exploitation (β high); the *decoupling* of direction noise from
   magnitude determinism is a genuinely non-obvious design point worth a paper figure.
6. **Future extensions:** hardware p-bit mapping (each coordinate = a physical stochastic unit, easy
   to map to fpga/CUDA kernels), GPU batch parallelization (all coordinates sampled in one op),
   quantum-inspired annealing, and plugging p-bit layers into larger models.

## #2 — Reproducible, Hardware-Aware Benchmark Harness (`pbit.bench`)
1. **Current behavior:** `main()` runs 3 benchmarks × 8 optimizers × 4 noise modes × N runs and
   emits CSV + 8 figures with a genuinely thoughtful metric set (AUC, escapes, noise degradation).
2. **Why valuable:** Not many small packages ship a *correct* stochastic-optimization benchmarking
   story. The escape-count and noise-robustness analysis are exactly what a reviewer wants, and
   "measure real hardware, label everything else as proxy" is a credibility differentiator. It makes
   every future PBit claim independently checkable.
3. **What must be rewritten:** turn `main()` into a `run_experiment(config) -> Report` with clean
   optimizer/fitness/noise interfaces; separate *measured* wall-time (distribution) from *proxy*
   FLOPs (explicitly labeled, computed by instrumented op counts — not a hardcoded guess);
   per-optimizer seeded RNG; drop plotting from the core path.
4. **Public API:**
   ```python
   from pbit.bench import Experiment, Rastrigin, Ackley, Rosenbrock, GaussianNoise
   from pbit.optim import PBitOptimizer, SGD, Adam

   ex = Experiment(
       functions=[Rastrigin(dim=2), Ackley(dim=2), Rosenbrock(dim=2)],
       optimizers={"pbit": PBitOptimizer(seed=1), "adam": Adam()},
       noise=[None, GaussianNoise(sigma=0.5)],
       max_iter=500, n_runs=10, seed=42,
   )
   report = ex.run()
   report.df()          # tidy long-format DataFrame
   report.to_csv("out/results.csv")
   report.metrics()     # AUC, iter-to-threshold, variance, escapes
   report.summary()     # console table
   ```
5. **Technically interesting:** correct *measurement methodology* for stochastic optimizers (per-run
   seeding, distribution reporting, separating measured vs proxied cost) — this is a research-grade
   contribution in itself and hard to get right.
6. **Future extensions:** track progress/compare across versions (regression), GPU/FLOPs instrumented
   counting, parameter sweeps, standardized result schema for publication, auto-report generation.

## #3 — Noise Robustness & Quantization Analysis (`pbit.bench.noise`)
1. **Current behavior:** gaussian/corruption/quantize gradient noise modes + fig2/7/8 analysis
   showing PBit's resilience (fig7 quantization sweep, fig8 degradation ratio).
2. **Why valuable:** PBit is *inherently* a stochastic/quantized-computation optimizer — this is the
   single most physically-honest claim it can make, and it ties directly to low-precision hardware
   (quantize to 1–4 bits). It differentiates PBit from SGD/Adam cleanly and motivates real hardware.
3. **What must be rewritten:** generalize noise injectors to a `Noise` protocol applying to gradients
   (sample-wise), add degradation-ratio and robustness-score as first-class metrics.
4. **Public API:**
   ```python
   from pbit.bench import QuantizeNoise, GaussianNoise, CorruptionNoise
   report.robustness_ratio()          # noisy_final / clean_final
   report.quantization_sweep(bits=[1,2,3,4,6,8,16,32])
   ```
5. **Technically interesting:** ties stochastic optimization to *hardware-precision* robustness — a
   credible bridge to Grunchie's hardware/FPGA interests.
6. **Future:** sweep train on quantized first-order updates; map directly to p-bit hardware precision
   limits; robustness as a scoring gate.

## #4 — (Conditional) P-Bit Attention / Stochastic Routing (research thread)
1. **Current behavior:** toy transformer with temperature-scaled softmax attention + entropy/sparsity
   metrics; an "experiments to try" list (p-bit edge sampling, dynamic/per-head temperature, Ising
   couplings, annealing).
2. **Why valuable:** interesting direction (thermo-stochastic attention as a routing mechanism), and
   it connects the optimizer idea to deep models — broadens the "p-bit compute" story beyond
   toy optimization. **However:** it is currently a toy, buggy, and not a product.
3. **What must be rewritten:** everything — separate module `pbit.attention`, real entropy-controlled
   stochastic routing, gradient-able temperature, tests.
   **Recommendation:** keep OUT of the v1 PyPI core. Promote to flagship only after it demonstrates
   genuine value in a research note / example. Rank #4 deliberately — it is the *highest-risk, least
   mature* item. Do not let it dilute the first release.

---

# 5. PyPI Product Vision

> If someone installs `pbit` from PyPI six months from now, what should they be able to do with it?

**30-second pitch:** "PBit is a NumPy-first library for *probabilistic* (p-bit) stochastic
optimization and reproducible benchmarking. Solve hard multimodal/non-convex problems with a
Boltzmann-driven binary-direction optimizer that is naturally robust to low-precision/quantized
gradients, and validate any optimizer's claims against clean, hardware-aware baseline benchmarks."

**Target users:**
- **Researchers** studying stochastic/probabilistic optimization, hardware-precision ML, p-bit /
  stochastic-computing (primary — Grunchie Labs research).
- **ML engineers** needing a robust, quantization-tolerant optimizer for low-precision edge/FPGA
  training.
- **Practitioners** who want a trustworthy comparison harness to evaluate their own optimizers.

**Primary use cases:**
1. `PBitOptimizer` as a drop-in optimizer for small/medium non-convex problems (classic benchmarks,
   and via a `ParamOptimizer` for array-valued params).
2. `pbit.bench` to run a reproducible, paper-quality optimizer comparison and export results.
3. Noise/quantization robustness studies.

**Package philosophy:**
- **Small, elegant, NumPy-first** core. One idea, done well, fully testable, sparse dependencies.
- **Measured ≠ proxy** — never conflate performance.
- **Reproducible by default** — seeds/hashable config, results carry metadata.
- **Infrastructure-optional** — artifacts/plots/backends are opt-in, never required to run.

**Scope boundaries / core API surface (small):**

```
pbit.core         sigmoid, bernoulli_bit, sigmoid_probability, schedules, config
pbit.optim        Optimizer protocol, PBitOptimizer, baselines (SGD/Adam/Momentum/RMSProp/Langevin/SimAnneal)
pbit.bench        Fitness, Noise, Metrics, Experiment/Report, run_experiment
pbit.bench.functions   Rastrigin, Ackley, Rosenbrock (+ analytical gradients)
pbit.bench.noise       Gaussian, Corruption, Quantize
pbit.bench.metrics     auc_loss, iter_to_threshold, final_variance, escapes, robustness
pbit.utils        rng helpers, timers, versioning
```

**What should NOT be included (v1):**
- The PyTorch attention transformer (promote later as `pbit.attention`, separate opt-in install).
- GPU/CUDA/Triton backends (no evidence yet they're needed; keep a thin seam).
- A zoo of 50 optimizers — only PBit + well-understood baselines.
- Advanced visualization/Web dashboarding.
- Reinforcement learning, graph/network simulators.

**Avoid:** turning into an unfocused collection of research experiments. Every module must serve the
two flagship pillars.

---

# 6. Proposed Architecture

```
Public API  (pbit/__init__.py, pbit.optim, pbit.bench)
    ↓
Core primitives  (pbit/core)
    ↓
Algorithms  (pbit/optim)
    ↓
Backend  (pbit/backends/numpy — seam only)
```

## 6.1 Proposed Layout

```
pbit/
├── __init__.py            # public exports; __version__
├── _version.py            # single source of version (from packaging)
├── config.py              # base dataclasses (SeedConfig, OptimizerConfig, ExperimentConfig)
├── core/
│   ├── __init__.py
│   ├── probability.py     # sigmoid, sigmoid_probability, bernoulli_bit, g_scale_normalize
│   ├── schedule.py        # linear_cooling, inverse_linear, plateau, adaptive(placeholder)
│   └── rng.py             # make_rng(seed) -> np.random.Generator, derive_seed
├── optim/
│   ├── __init__.py        # PBitOptimizer, baselines
│   ├── base.py            # Optimizer protocol + OptimizerResult
│   ├── pbit.py            # PBitOptimizer (+ step-size + transition + schedule strategies)
│   └── baselines.py       # SGD, Momentum, Adam, RMSProp, Langevin, SimAnneal
├── bench/
│   ├── __init__.py
│   ├── fitness.py         # Fitness base (func + grad), helpers
│   ├── functions.py       # Rastrigin, Ackley, Rosenbrock (+ analytic grads)
│   ├── noise.py           # Noise protocol: Gaussian, Corruption, Quantize
│   ├── metrics.py         # auc_loss, iter_to_threshold, final_variance, escapes, robustness
│   ├── experiment.py      # Experiment, BenchmarkRunner, ExperimentConfig
│   ├── report.py          # Report: tidy df, CSV/JSON export, metadata, summary
│   └── plot.py            # OPTIONAL matplotlib figures (lazy import)
├── backends/
│   ├── __init__.py        # Backend = numpy (only); export ACCESS pattern
│   └── numpy_backend.py   # tiny indirection (sample, randn) — seam for future rust/triton
└── utils/
    ├── __init__.py
    ├── timing.py          # perf_counter timer, distribution reporting
    └── io.py              # atomic CSV/JSON writes, metadata hashing
```

### Module responsibilities & source mapping

| New module | Responsibility | Key items | Moves from |
|-----------|----------------|-----------|-----------|
| `core/probability.py` | Math primitives of p-bit transitions | `sigmoid`, `sigmoid_probability`, `bernoulli_bit`, `normalize_scale` | inline logic in PBit (test_2.py:145-147) |
| `core/schedule.py` | Temperature scheduling | `linear_cooling` = `min(β0(1+t/τ), βmax)` | test_2.py:144 |
| `core/rng.py` | Seeded RNG isolation | `make_rng`, per-run stream | test_2.py:36,724 |
| `optim/base.py` | Optimizer protocol + result | `Optimizer`, `step(x, grad, t)`, `OptimizerResult` | all optimizers |
| `optim/pbit.py` | Flagship p-bit optimizer | `PBitOptimizer` + strategy hooks | test_2.py:121-151 |
| `optim/baselines.py` | Reference optimizers | SGD/Adam/Momentum/RMSProp/Langevin/SimAnneal | test_2.py:152-212 |
| `bench/functions.py` | Fitness functions + grads | Rastrigin/Ackley/Rosenbrock | test_2.py:52-92 |
| `bench/noise.py` | Gradient noise | Gaussian/Corruption/Quantize | test_2.py:97-115 |
| `bench/metrics.py` | Metrics | auc, threshold, variance, escapes, robustness | test_2.py:268-322 |
| `bench/experiment.py` | Orchestrator | `Experiment`, `BenchmarkRunner` | run_optimizer/run_all/main (test_2.py:232-262,491) |
| `bench/report.py` | Results + export | tidy df, CSV/JSON, metadata | main() CSV writer (test_2.py:519-528) |
| `bench/plot.py` | Optional figures | the 8 fig helpers | test_2.py:329-489 (lazy-guarded) |
| `backends/numpy_backend.py` | Future backend seam | `sample`, `randn` | (NEW, minimal) |
| `utils/timing.py` / `io.py` | Timing + IO | timers, atomic writes | test_2.py:20,224 |

**Dependency flow:** `pbit.optim` → `pbit.core`; `pbit.bench` → `pbit.core` + `pbit.optim` + `pbit.utils`;
`pbit/__init__` re-exports the public API. `backends` is referenced only by `core.rng`/`core.probability`
through a tiny seam so nothing today changes.

**Avoid over-abstraction:** No ABCs everywhere. Use a lightweight `Protocol`, plain functions for
schedules/fitness/noise, dataclasses for config/results. `backends` stays a *seam* (one tiny
indirection), not a plugin system.

---

# 7. API Design Plan

```python
# 1. Core p-bit primitives (rarely used directly but public)
import pbit
from pbit.core import sigmoid, sigmoid_probability, bernoulli_bit, linear_cooling

# 2. The optimizer
from pbit.optim import PBitOptimizer

opt = PBitOptimizer(lr=0.01, beta0=2.0, tau=150.0, beta_cap=50.0, seed=1)
for t in range(500):
    g = grad(x)
    x = opt.step(x, g, t)          # returns new params
state = opt.state()                # dict of beta, step_count, hyperparams (for reporting)

# 3. Baselines share the same protocol
from pbit.optim import Adam, SGD, Momentum, RMSProp, Langevin, SimAnneal
adam = Adam(lr=0.01).reset()  # protocol: reset(), step(x, grad, t)

# 4. Benchmarking
from pbit.bench import Experiment, Rastrigin, Ackley, Rosenbrock, GaussianNoise, QuantizeNoise
from pbit.optim import PBitOptimizer, Adam, SGD

config = {
    "functions": [Rastrigin(dim=2), Ackley(dim=2), Rosenbrock(dim=2)],
    "optimizers": {"pbit": PBitOptimizer(beta0=2.0, tau=150), "adam": Adam(), "sgd": SGD()},
    "noise": [None, GaussianNoise(sigma=0.5), QuantizeNoise(bits=4)],
    "max_iter": 500, "n_runs": 10, "seed": 42,
}
report = Experiment(config).run()

report.df()                       # tidy long-format DataFrame
report.to_csv("out/res.csv"); report.to_json("out/res.json")
report.metrics()                  # dict of metrics per (function, optimizer, noise)
report.robustness_ratio()         # noisy/clean final loss
report.quantization_sweep(bits=[1,2,3,4,8,16,32])
print(report.summary())           # console table
```

Naming conventions:
- **Configs:** `ExperimentConfig`, `OptimizerConfig`, `SeedConfig` — dataclasses, frozen where safe.
- **Results:** `OptimizerResult` (optimizer-run), `Report` (whole experiment), `MetricResult`.
- **Functions/schedules/noise:** lowercase functions implementing `Protocol`s for composability.
- **Import paths are short and stable:** `pbit.optim`, `pbit.bench`, `pbit.core`, `pbit.utils`.

---

# 8. Benchmarking System Redesign

## 8.1 Current problems (methodology)
1. **Measured vs proxy conflation — the headline issue.** The `efficiency score` =
   `1/(time·iters)·1e3` mixes `time` (measured, hardware/implementation-dependent) with iters
   (algorithmic). PBit is penalized purely because its per-step cost is higher in pure NumPy. The
   FLOPs number is a **hardcoded constant** (10000 for everyone) and therefore meaningless.
2. **One LR for all optimizers.** Conclusions are tuned-to-PBit by construction (PBit default
   LR=0.01 vs Adam default on same LR). Fair comparison needs either per-optimizer tuned LR, or an
   explicit, reported "default-parameter, no-tuning" framing.
3. **`auc_loss` is scale-dependent** — across Rosenbrock (loss ~10s) vs Rastrigin (loss ~3) they are
   not comparable; the metric is only valid *within* one function. Must be explicit.
4. **`iter_to_threshold` uses a fixed threshold=1.0** — for Rosenbrock (values ~11) most optimizers
   report 500 (never), while PBit hits 51; the fixed constant is arbitrary and should be
   configurable/suite-relative.
5. **`count_escapes` parameters** (window=20, delta=0.05, drop=5%) are magic constants; fragile and
   undocumented. Redesign as a documented, parameterized method with a clear definition.
6. **Global RNG / shared stream** breaks strict reproducibility and cross-run independence (see §9).
7. **Non-ASCII CSV header** (`Iter→threshold`) — breaks portable parsing (Excel/SomeCSV). Use ASCII.

## 8.2 Reusable framework

```python
class Fitness(Protocol):
    name: str
    dim: int
    def __call__(self, x) -> float: ...
    def grad(self, x) -> np.ndarray: ...

class Noise(Protocol):
    name: str
    def __call__(self, g: np.ndarray) -> np.ndarray: ...   # applied to gradient
    # evaluate/fit hooks for adaptive noise later

class Optimizer(Protocol):
    def reset(self, rng) -> None: ...
    def step(self, x, grad, t) -> np.ndarray: ...

class Experiment:
    def __init__(self, config: ExperimentConfig): ...
    def run(self) -> Report: ...
```

Experiment config (frozen dataclass): functions, optimizers (dict name→factory), noise list,
`max_iter`, `n_runs`, `seed`, `threshold`, `clip`, `out_dir`, `save_artifacts`.

Benchmark runner semantics:
- For each (function, noise, optimizer): create optimizer via factory, **create a fresh RNG derived
  deterministically per `(seed, function, noise, optimizer, run)`**, run `max_iter` steps, record
  loss history, wall-time per run. Report **distribution** (mean/median/std/min/max) not just mean.
- Start positions drawn from the derived RNG (reproducible per cell).

## 8.3 Reproducibility & schema
Result long-format schema (each row = one optimizer-run):

```
experiment_id, pbit_version, seed, config_hash,
function, dim, noise, optimizer,
run_id, iter_t, loss_t, wall_time_s,
```

Summary metrics table:

```
experiment_id, function, noise, optimizer,
iter_to_threshold, auc_loss, final_loss_mean/median/std,
escape_count, robustness_ratio, wall_time_mean/std, flop_count(proxy, labeled)
```

Export: `CSV` (ASCII) + `JSON` (with full metadata incl. `pbit.__version__`, `np.__version__`,
config, seed, git commit if available, hardware string). `to_csv`/`to_json`.

## 8.4 Metric definitions (final)
- **iter_to_threshold(f, τ)** — first `t` with mean loss < τ; `max_iter` if never (τ configurable).
- **auc_loss(f)** — trapezoid over iterations normalized by iteration count (documented as
  function-relative only).
- **final_loss_mean / -median / -std** — across runs at final iteration (prefer median+std for
  robustness to outliers).
- **escape_count(f, window, delta_threshold, drop)** — parameterized plateau-escape detector
  (documented).
- **robustness_ratio(noisy, clean)** = `final_loss_noisy / final_loss_clean` (≥1 → degradation;
  band around 1 = robust). This is the cleanest noise metric — keep it prominent.
- **wall_time (measured)** — distribution; *hardware performance*, always labeled "measured".
- **flop_count (proxy)** — **redesigned**: computed by instrumented operation counters per optimizer
  (count FLOPs of the math actually performed), labeled explicitly as a **proxy/estimated** cost and
  NEVER mixed into an aggregate "score" with measured wall time.

**Distinguish clearly (documented in the report):**
- **Measured hardware performance:** wall-time, memory.
- **Proxy/estimated performance:** FLOP counts, "efficiency" aggregates. All proxy fields carry a
  `kind="proxy"` tag; measured fields carry `kind="measured"`. This is a first-class API contract.

## 8.5 Comparison / report generation
`Report` produces a tidy DataFrame + summary table; optional `report.plot(...)` (matplotlib, lazy)
regenerates the 8 figures from v2. The 8 figures are worth preserving (they are high quality) as the
default report gallery, decoupled from the run path.

---

# 9. Reproducibility & Scientific Rigor

**Core principle:** *a deterministic function of the experiment config.* Two researchers running on
the same `ExperimentConfig` + same `pbit` version must produce bit-identical loss histories (given a
deterministic backend).

Design:
- **RNG manager** (`core/rng.py`): a single master `seed`; derive per-cell seeds via a stable
  `seed = hash((master_seed, function, noise, optimizer, run))` → `np.random.default_rng(seed)`.
  No module-global `np.random.seed` calls in the library. Optimizers/fitness/noise receive the RNG or
  seed explicitly — **never** draw from the global stream.
- **Config hash:** `config_hash = sha256(canonical_json(config))` included in every result row and
  artifact name.
- **Version pinning:** record `pbit.__version__`, numpy version, python version, platform, and (if
  git-available) commit hash in JSON metadata so two users who get different numbers can see *why*:
  different config, version, or hardware (last one only affects measured time, not loss).
- **Determinism caveat on measured time:** loss histories deterministic across machines; wall-times
  are not (report as separate measured column, never asserted for equality).
- **Fitness/grad determinism:** finite-difference grad for Ackley uses a deterministic eps pert; no
  randomness inside fitness.
- **Documented "why results differ" workflow:** `Report.metadata` exposes every factor; a
  `pbit bench compare a.json b.json` helper (or documented script) diffs config hashes & versions and
  reports the first differing field. This directly answers the "why did ours differ" question.

---

# 10. Testing Strategy

## 10.1 Prioritization (most urgent first)
1. **Mathematical correctness of primitives** (`core/probability.py`, `core/schedule.py`) — no tests
   exist today; highest risk.
2. **PBit update rule** must reproduce the exact v2 math (`lr·|g|·σ`), incl. random-seed-for-value
   checks.
3. **Baseline optimizers** — Adam/SGD etc. are known-good; test they converge on convex toy problems
   and match reference behavior on seeded inputs.
4. **Reproducibility** — same config → identical histories; per-run RNG isolation.
5. **Metrics** — hand-computed expected values for auc/iter-to-threshold/escapes on tiny synthetic
   curves.
6. **Benchmark functions** — known minima at x=0 (Rastrigin/Ackley f=0), Rosenbrock min=0 at (1,1);
   gradient vs finite-diff agreement.
7. **Noise injectors** — statistical properties (mean/variance), quantization round-trip.
8. **Experiment/Report** — schema shape, CSV/JSON round-trip.

## 10.2 Architecture
- `pytest`, `numpy.testing` for array equality (`assert_allclose`).
- **Deterministic (seeded) tests** for stochastic pieces: fix seed, assert exact array output (catch
  regressions in RNG usage).
- **Property/statistical tests**: e.g., `bernoulli_bit` mean ≈ p within tolerance on large N;
  `sigmoid_probability` monotonicity.
- **Convergence tests**: PBit converges on Rosenbrock in finite iters (regression gate — the claim
  from §4.1#1).
- **Edge cases**: zero-gradient (`g_scale+ε` path), constant gradient (quantize guard `gmax==gmin`),
  dim=1, empty arrays, NaN handling, clip bounds.
- **Performance smoke test**: a marked (`@pytest.mark.perf`) coarse threshold to catch accidental
  Python-loop regressions in hot paths (not micro-benchmarks).
- **Regression test** that `pbit bench demo` output is stable across runs (config-hash gated).

Urgent gap: today there are **zero tests** and zero package importability. First test = "import
works" + "primitive math correct".

---

# 11. Documentation Strategy

```
README.md            (30-second pitch + quickstart + one figure)
docs/
├── api/             (per-module API reference — generated from docstrings)
├── tutorials/
│   ├── 01_optimizer_basics.ipynb
│   ├── 02_run_a_benchmark.ipynb
│   ├── 03_noise_robustness.ipynb
│   └── 04_reproducibility.ipynb
├── benchmarks.md    (metric definitions, measured-vs-proxy, fair-comparison methodology)
├── research/        (research notes: the p-bit math, annealing, hardware mapping)
└── architecture.md  (module map + dependency flow + backend seam)
examples/
├── demo_optimizer.py
├── run_benchmark.py
├── noise_sweep.py
└── compare_two_reports.py
CHANGELOG.md
LICENSE
```

**README first 30 seconds must communicate:**
1. What it is (one sentence): "PBit: NumPy-first probabilistic (p-bit) stochastic optimization +
   reproducible benchmarking."
2. One 5-line code snippet using `PBitOptimizer` and `Experiment`.
3. One compelling figure (Rosenbrock convergence showing PBit beating Adam — the data already proves
   this: iter 51 vs 500).
4. Why it exists (the p-bit hardware story / measured-vs-proxy honesty).
5. Three badges: `pip install pbit`, build pass, version.

**3–5 example projects (why the package exists):**
1. **`examples/demo_optimizer.py`** — drop-in `PBitOptimizer` on the 3 classic functions; the
   "hello world."
2. **`examples/run_benchmark.py`** — full reproducible paper-style comparison (8 optimizers × 3
   functions × 4 noise modes) → CSV/JSON/figures. Mirrors the current `test_2.py` value.
3. **`examples/noise_sweep.py`** — quantization bit-width sweep (the §4#3 story) showing PBit's
   robustness vs Adam.
4. **`examples/compare_two_reports.py`** — the reproducibility workflow: diff two result files and
   explain discrepancies (config/version/hardware attribution).
5. **(future)** `examples/pbit_attention.py` — stochastic-attention research demo (excluded from v1
   core; shipped as example once mature).

---

# 12. Packaging & Release Plan

## 12.1 `pyproject.toml` (PEP 621)
- **Build backend:** `setuptools` (or hatchling — recommend **hatchling** for cleanliness).
- **Metadata:** name `pbit`, version from single source (`pbit/_version.py` via dynamic), description,
  readme, authors (± Grunchie Labs), license (recommend **MIT**; confirm with Grunchie), keywords,
  classifiers (Programming Language :: Python :: 3, Topic :: Scientific/Engineering, Development
  Status :: 3 – Alpha for 0.x).
- **Dependencies (runtime):** `numpy>=1.23` (only hard dep). `matplotlib` and `pandas` are
  **optional** (`[pbit-plots]`, `[pbit-tables]` extras). `torch` is explicitly NOT a core dep
  (only in examples/future attention extra).
- **Optional extras:** `bench` (matplotlib, pandas), `dev` (pytest, ruff, mypy), `docs` (sphinx/mkdocs).
- **Python:** `requires-python = ">=3.9"` (verify against 3.14; library should be 3.9–3.14 compatible).
- **Packages:** find `pbit`.

## 12.2 Build & distribution
- `python -m build` → both **wheel** (`pbit-*.whl`) and **source dist** (`sdist`).
- Publish via `twine upload dist/*` to TestPyPI then PyPI.
- `.gitignore` for `dist/`, `build/`, `*.egg-info/`, `__pycache__/`, `pbit_benchmark_results/`.

## 12.3 Version strategy (calver not needed; semver)
- **`0.1.0`** — First usable *internal* release: core p-bit optimizer + benchmark single-function,
  tests pass, basic README. Not yet public-PyPI full.
- **`0.2.0`** — Full benchmark framework (multi-function, noise, report/export), reproducibility,
  figures, examples. Candidate for TestPyPI.
- **`0.3.0`** — Docs complete, packaging polished, CI green, performance Phase-1 done. **First PyPI
  `0.3.0` release.**
- **`1.0.0`** — Stable public API (no breaking changes without deprecation), extended test matrix,
  optional backend (Rust/CUDA) if justified, attention module if matured. All flagships stable.
- Pre-release tags during dev: `0.3.0rc1` etc. Adopt **CalVer fallback** if Grunchie prefers.

## 12.4 CI
- GitHub Actions: matrix (3.9…3.14 / 3.14), ubuntu-latest (plus windows-latest for parity since dev
  is on Windows).
- Jobs: lint (ruff), typecheck (mypy, strict on `pbit/core` + `pbit/optim`), test (pytest incl.
  coverage), build (wheel+sdist), publish-on-tag (twine, PyPI token secret).
- Nightly benchmark regression gate (optional, marked perf).

## 12.5 Lint/type/test tooling
- **ruff** (lint + format), **mypy** (typed core), **pytest + coverage**, **build**, **twine**.

## 12.6 License & changelog
- **LICENSE:** MIT (or Apache-2.0 if Grunchie prefers patent/frank for hardware). Decide with Grunchie.
- **CHANGELOG.md:** keep-a-changelog format; each release.

---

# 13. Migration Plan

`CURRENT FILE` → `NEW LOCATION` → `WHAT CHANGES` → `WHY`

| Current | New location | Changes | Why |
|---------|--------------|---------|-----|
| `test_2.py` (PBit, lines 121-151) | `pbit/optim/pbit.py` | Extract update rule into `PBitOptimizer` with strategy hooks + explicit RNG; keep exact `lr·|g|·σ` math | Flagship #1; needs testable/seeded form |
| `test_2.py` (baselines 152-212) | `pbit/optim/baselines.py` | Unify under `Optimizer` protocol; `reset()`; remove dead `x_best/f_best`; drop `EvoStrat` from v1 (or hide behind `bench.baselines`) | Standard refs; protocol uniformity (EvoStrat's `set_fns` breaks the protocol) |
| `test_2.py` (functions 52-92) | `pbit/bench/functions.py` | Wrap as `Rastrigin/Ackley/Rosenbrock` classes; add analytic Ackley gradient (drop loop FD) | Testability, speed, fairness |
| `test_2.py` (noise 97-115) | `pbit/bench/noise.py` | `Noise` protocol; keep math; `QuantizeNoise(bits)` | §4#3 flagship; composability |
| `test_2.py` (metrics 268-322) | `pbit/bench/metrics.py` | Keep auc/threshold/variance; redesign `escape` (parms), add median, robustness_ratio; ASCII names | Methodology fixes (§8) |
| `test_2.py` (runner 232-262, main 491) | `pbit/bench/experiment.py` + `report.py` | `Experiment(config).run() -> Report`; per-run seeded RNG; separate measured vs proxy | Scalability, reproducibility |
| `test_2.py` (plots 329-489) | `pbit/bench/plot.py` | Preserve the 8 figures; lazy matplotlib import; detached from run path | Keep strong figures; headless-core |
| `test_2.py` (config/globals) | `pbit/config.py` + `core/rng.py` | Dataclasses; `make_rng`; drop module-global state | Rigor |
| `Untitled-1.py` | **DELETE** (ideational) | — | 93% duplicate of test_2.py; divergence risk. Archive content in research notes only |
| `pbit_attention.py` + notebook | `examples/` (future `pbit/attention/`) | Rewrite with entropy-controlled stochastic routing AFTER v1; NOT core | Toy, buggy (`p_bit_weight=`), unfocused for release |
| `pbit_benchmark_results/*` | Moved under `out/` (generated), gitignored | Regenerate via new `Experiment` | Never commit outputs as source |

**Reuse vs rewrite decision:**
- **REUSE as-is (copy math):** PBit update rule, baseline formulas, benchmark functions, noise math,
  auc/threshold/variance metrics, the 8 plot functions. Mathematically sound; do not rewrite working
  math without reason.
- **REWRITE the scaffolding:** packaging, config, RNG isolation, metrics redesign (escape/robustness),
  measured-vs-proxy, experiment/report orchestration, docs/tests.

---

# 14. Development Roadmap

### Phase 0 — Audit (0.5–1 day)
- **Objectives:** confirm inventory; decide flagships; lock scope. *(This document is Phase 0's
  output.)*
- **Files affected:** none.
- **Done:** scope sign-off with Grunchie Labs.

### Phase 1 — Core rewrite (1–2 days)
- **Objectives:** package skeleton + core primitives + PBit optimizer, tested.
- **Files:** `pyproject.toml`, `pbit/__init__.py`, `pbit/core/*`, `pbit/optim/pbit.py`, tests.
- **Tasks:** create package, implement `core.probability/schedule/rng`, port PBit v2 rule exactly,
  seedable, first tests.
- **Deps:** none (foundation).
- **Done:** `pbit.optim.PBitOptimizer` passes math + determinism tests; package importable.

### Phase 2 — Optimizer API (1 day)
- **Objectives:** uniform `Optimizer` protocol + all baselines.
- **Files:** `pbit/optim/base.py`, `baselines.py`, tests.
- **Tasks:** protocol, `reset/step`, port baselines, converge-on-convex tests.
- **Deps:** Phase 1.
- **Done:** SGD/Adam/Momentum/RMSProp/Langevin/SimAnneal work under one interface.

### Phase 3 — Benchmark framework (2–3 days)
- **Objectives:** `Experiment`/`Report` + metrics + noise + export + figures.
- **Files:** `pbit/bench/*`, tests.
- **Tasks:** functions/noise/metrics/experiment/report; measured-vs-proxy tagging; CSV(ASCII)/JSON;
  port the 8 plots to optional `plot.py`.
- **Deps:** Phase 1–2.
- **Done:** `Experiment(...).run() -> Report.to_csv/json/df` reproduces the CSV table with correct,
  honest metrics.

### Phase 4 — Performance (2–3 days)
- **Objectives:** Phase-1 CPU optimization (§8).
- **Files:** `pbit/core`, `pbit/optim/pbit.py`, profiling.
- **Tasks:** vectorize hot loops, precompute schedules, reuse buffers, profile.
- **Deps:** Phase 1–3.
- **Done:** profiled; PBit per-step cost within reasonable constant factor of Adam; benchmark
  measured-time column meaningful.
*(Phase 2/3 Rust & Phase 3/4 GPU are post-1.0; see §8 performance roadmap.)*

### Phase 5 — Testing (1–2 days, ongoing)
- **Objectives:** full suite from §10.
- **Files:** `tests/*`.
- **Done:** `pytest` green, coverage targets met, reproducibility + convergence regression gates in CI.

### Phase 6 — Documentation (2 days)
- **Objectives:** README, tutorials, benchmarks methodology doc, examples, API docs.
- **Files:** `README.md`, `docs/*`, `examples/*`, `CHANGELOG.md`.
- **Done:** README first-30-seconds strong; 4 examples run; benchmark methodology documented.

### Phase 7 — Packaging (1 day)
- **Objectives:** polished build, extras, CI.
- **Files:** `pyproject.toml`, CI workflow, LICENSE.
- **Done:** `python -m build` produces wheel+sdist; CI green on matrix.

### Phase 8 — Release (0.5 day)
- **Objectives:** publish.
- **Files:** version bump, CHANGELOG, twine upload.
- **Done:** `pbit 0.3.0` on PyPI; `pip install pbit` works.

---

# 15. Prioritization

### MUST HAVE (first PyPI release `0.3.0`)
1. `pbit.core` primitives + `PBitOptimizer` (exact v2 math) + seedable RNG.
2. `pbit.optim` uniform protocol + baselines (SGD/Adam/Momentum/RMSProp/Langevin/SimAnneal).
3. `pbit.bench` functions (Rastrigin/Ackley/Rosenbrock) + noise + metrics.
4. `Experiment`/`Report` with measured-vs-proxy separation + CSV/JSON export.
5. Reproducibility (config hash, per-run seeds, version pinning).
6. Unit tests: primitives, PBit rule, reproducibility, metrics, functions.
7. `pyproject.toml` + wheel/sdist + LICENSE + README + minimal CI.
8. Drop `EvoStrat` from the release optimizer set (keep behind `bench.baselines` or delete).

### SHOULD HAVE (strongly useful, not blocking)
- The 8-figure report gallery (`pbit.bench.plot`) preserved/moved.
- `robustness_ratio` + quantization-sweep as first-class report methods (§4#3 shipping).
- Phase-1 performance pass (measured time meaningful).
- Full CI matrix (multiple Python versions; Windows + Linux).
- mypy typing on core/optim.
- Tutorial notebooks + `examples/`.

### NICE TO HAVE (post-1.0)
- Performance Phase-2 Rust/native acceleration (only if profiling shows Python overhead dominates and
  it matters to users).
- Backend seam filled: numba/C extension for batch `bernoulli_bit`.
- `pbit.attention` stochastic-attention module (once it demonstrates real value; research thread).
- Adaptive temperature schedules driven by entropy.
- CLI (`pbit bench ...`) for headless reproducibility.

### DO NOT BUILD YET (complexity without enough value / marketing-flavored)
- **GPU/CUDA/Triton** backend at v1 — no evidence it's needed; it would require a different
  architecture and distro complexity. Revisit after profiling shows a real bottleneck. (Performance
  §Phase-3 explicitly postpones.)
- A large optimizer zoo.
- A Web dashboard / heavy visualization framework.
- RL or network/graph simulators.
- Hardware/FPGA tooling or emulator (document the mapping in research notes only).

---

# 16. Major Technical Risks

1. **RNG/reproducibility fragility** (highest). Getting the derived-seed scheme wrong reintroduces
   the global-state bug. Mitigate: dedicated `core/rng.py`, test-locked seeds, config hash.
2. **Fair-comparison bias** (§8.1-2). One-LR-fits-all makes results tuned-to-PBit. Mitigate:
   documented default-parameters framing, per-optimizer tuned LR option, honest methodology doc.
3. **PBit performance constant-factor gap.** Naive NumPy PBit is slower per-step than Adam; the
   "efficiency score" would unfairly penalize it. Mitigate: measure-and-label honestly, Phase-1
   vectorization, never merge measured+proxy.
4. **Scope creep from the attention thread.** Promoting `pbit.attention` too early would ship an
   unfocused toy. Mitigate: keep it out of v1 core; gate on demonstrated value.
5. **Version/platform skew** (3.9–3.14, Windows dev vs CI Linux). Mitigate: CI matrix, `requires-python`,
   pinning numpy floor.
6. **`escape_count` fragility.** Parameter-sensitive metric can produce misleading "findings."
   Mitigate: document + parameterize + treat as exploratory, not headline.
7. **One maintainer / long tail.** Mitigate: lock scope (small API), good docs/tests so another agent
   or engineer can execute this plan phase-by-phase.

---

# 17. Final Recommended Architecture

- **NumPy-only core**, thin (no plugin framework).
- Three-level flow: **Public API → core primitives → algorithms**, with **backends** a *seam* not a
  product.
- Commit to **two flagships**: `pbit.optim.PBitOptimizer` (research core) and `pbit.bench`
  (reproducible, measured-vs-proxy-honest harness). Everything else is scaffolding.
- **Explicitly exclude** GPU, attention (core), optimizer zoo, dashboards at v1.
- **Reproducibility by design**, not retrofit (derived RNG, config hash, version pinning, ASCII
  schema).
- Ship **`0.3.0`** to PyPI as the first public release; reserve `1.0.0` for a stable API + optional
  acceleration.

This plan is self-contained: a fresh engineer/agent can execute Phase 0→8 from this document without
rederiving the project. The only code worth preserving *unchanged* is the PBit v2 update math and the
8-figure plotting logic; everything else is reconstructed into a clean library around those assets.
