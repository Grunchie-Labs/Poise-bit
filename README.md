# Pbit

> Probabilistic p-bit optimization, low-precision gradient robustness, and
> reproducible optimizer benchmarking.

`pbit` is a small, NumPy-first (PyTorch-optional) library for **probabilistic
binary-direction (p-bit) optimization**. It is designed for a world where
optimizers increasingly operate under low precision, noisy gradients, and
quantization — and it ships an **honest benchmark harness** that separates
measured hardware time from declared proxy cost, and always reports
best-so-far behavior.

## Why p-bit optimization?

Modern optimization is moving toward low-precision, noisy, energy-constrained
hardware. `pbit` is naturally stochastic and binary-directional: each
parameter update is a Bernoulli decision `σ ∈ {−1, +1}` whose probability
follows a Boltzmann distribution, modulated by an inverse‑temperature
annealing schedule. The result is an optimizer that is structurally at home
with quantized/noisy gradients, while still behaving like gradient descent as
it cools.

## Quickstart

```python
from pbit.optim import PBitOptimizer

opt = PBitOptimizer(lr=0.01, beta0=2.0, tau=150.0, seed=42)
for t in range(500):
    g = grad(x)
    x = opt.step(x, g, t)
```

Low-precision stress test across optimizers:

```python
from pbit.bench import Experiment, Rastrigin, QuantizeNoise
from pbit.bench.specs import OptimizerSpec
from pbit.optim import Adam, PBitOptimizer, SGD

report = Experiment({
    "functions": [Rastrigin(dim=2)],
    "noises": [None, QuantizeNoise(bits=1, stochastic=True)],
    "optimizers": [
        OptimizerSpec("pbit", lambda: PBitOptimizer(lr=0.05, tau=300, step_size="floor", floor=1e-3)),
        OptimizerSpec("adam", lambda: Adam(lr=0.05)),
        OptimizerSpec("sgd", lambda: SGD(lr=0.05)),
    ],
    "max_iter": 500, "n_runs": 10, "seed": 42,
}).run()

print(report.summary())
report.to_csv("out/results.csv")
report.to_json("out/results.json")
```

Optional PyTorch optimizer:

```bash
pip install "pbit[torch]"
```

```python
from pbit.torch import PBitTorchOptimizer

optimizer = PBitTorchOptimizer(model.parameters(), lr=1e-3, tau=1000)
```

## Install

```bash
pip install pbit            # NumPy core
pip install "pbit[bench]"   # + matplotlib / pandas for report helpers
pip install "pbit[torch]"   # + PyTorch optimizer wrapper
```

Requires Python >= 3.11. Runtime dependency: `numpy`.

## Highlights

- `pbit.optim.PBitOptimizer` — the probabilistic binary-direction optimizer,
  with `proportional`, `constant`, and `floor` step-size modes and
  thermodynamic diagnostics (`beta`, flip probability, entropy).
- `pbit.optim` baselines — SGD, Momentum, Adam, AdamW, RMSProp, SignSGD, Lion,
  Langevin, and ask/tell evolution-strategy & simulated annealing.
- `pbit.bench` — reproducible `Experiment`/`Report`, best-so-far histories,
  success-rate & median-hit-time metrics, function-specific domains, and an
  AI/hardware-relevant noise suite (`QuantizeNoise`, `SignNoise`, `ClipNoise`,
  `GaussianNoise`, `CorruptionNoise`).
- `pbit.torch.PBitTorchOptimizer` — optional PyTorch integration.
- `pbit.bench.compare_reports` — explain why two runs differ.

## What this is / is not

`pbit` is **not** an LLM optimizer, a hardware emulator, or a GPU-backed
library. It is a focused, NumPy-first research/engineering tool: the p-bit
optimizer and an honest way to measure when (and when not) it helps.

## License

Apache-2.0 — the explicit patent grant is intentional for
hardware-adjacent work.
