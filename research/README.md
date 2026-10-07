# Research objective

This repository holds a research project about p-bit optimization. The
objective is to state, as precisely as the evidence allows, what p-bit
optimization does and does not buy on low-precision noisy gradients, and to
show the measurement that supports each statement.

## Research question

A p-bit is a stochastic binary unit, so an optimizer built on p-bit dynamics
should tolerate quantized and noisy gradients better than a deterministic
method. The idea is plausible and partly motivated by the physics of the
hardware. Whether it holds in practice is an empirical question, and the
previous results in this repository answered it without reporting uncertainty.

The question this project asks:

> Under a measurement that cannot manufacture its own result, does p-bit
> optimization show an advantage on low-precision gradients, and where does it
> not?

The question is deliberately narrow. It can be answered either way, and the
answer is recorded in [`CLAIMS.md`](CLAIMS.md) whether or not it flatters the
method.

## What counts as an answer here

A claim counts as answered only when all of the following hold:

1. The statistic and the hypothesis are named before the measurement.
2. Every reported number comes from a script in `experiments/`, recorded in
   `results/`, never transcribed by hand into prose.
3. A mean is never reported without its spread and a confidence interval.
4. Optimizers are tuned on a declared grid on clean cells, the full grid is
   published, and evaluation runs on seeds the search never saw.
5. Comparisons are paired on shared random streams, so start position, noise
   perturbation, and private coins cancel within a run.
6. When the evidence does not separate two conditions, the claim is recorded
   as unresolved rather than argued in either direction.

Rule 6 did real work: several registered claims did not resolve, and one
resolved against the method.

## Layout

```
research/
├── README.md                  this file
├── METHOD.md                  measurement rules and why they exist
├── CLAIMS.md                  every claim, its verdict, and the numbers
├── THEORY.md                  why PBit is robust to some noise, and when not
├── RELATED.md                 literature positioning and verified citations
└── experiments/
    ├── tune_and_evaluate.py   tunes every optimizer, evaluates on held-out seeds
    ├── claims.py              computes every claim from the recorded series
    ├── power.py               subsample power curve and null check
    ├── robustness.py          verdict and direction stability across master seeds
    ├── mnist.py               real-task replication with an MLP on MNIST
    ├── clip_confound.py       measures the quantization-cell clipping confound
    └── verify_docs.py         checks CLAIMS.md against the recorded results

results/
├── tuned.json                 tuning grids, selected configs, eval series
├── tuned_seed1.json           seed replications (tuning unchanged)
├── tuned_seed2.json
├── claims.json                machine-readable claim verdicts
├── claims_seed1.json
├── claims_seed2.json
├── robustness.json            per-claim stability across master seeds
├── mnist.json                 MNIST real-task results
├── power.json                 subsample power analysis
├── clip_confound.json         clip-confound measurements
├── leaderboard.csv            per-cell mean terminal loss with sd, all optimizers
└── withdrawn.json             figures quoted from the withdrawn prior report
```

The library lives in `pbit/` and the research layer reads it without modifying
it; changes to measurement behaviour belong in the library and are covered by
tests in `tests/`.

## Reproducing

From the repository root:

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
python research/experiments/tune_and_evaluate.py
python research/experiments/claims.py
python research/experiments/power.py
python research/experiments/clip_confound.py
python research/experiments/mnist.py
python research/experiments/verify_docs.py
```

Seed replication (tuning is deterministic and unchanged; only eval re-runs):

```bash
python research/experiments/tune_and_evaluate.py --eval-seed 1 --out results/tuned_seed1.json
python research/experiments/tune_and_evaluate.py --eval-seed 2 --out results/tuned_seed2.json
python research/experiments/claims.py --tuned results/tuned_seed1.json --out results/claims_seed1.json
python research/experiments/claims.py --tuned results/tuned_seed2.json --out results/claims_seed2.json
python research/experiments/robustness.py
```

Everything is deterministic: randomness comes from SHA256-derived streams keyed
on the master seed, so the same commands reproduce the same numbers.
`results/tuned.json` records the software versions used for the checked-in
numbers. `verify_docs.py` checks every number written in `CLAIMS.md` against
the recorded JSON, and runs as part of the test suite; during writing it caught
two hand-computed Holm-adjusted p-values that did not match the recorded runs.

Expected runtime is about four minutes for `tune_and_evaluate.py`, ten minutes
for `mnist.py`, and seconds for the rest on a laptop.

## Status

Eight claims are registered across 17 comparisons in one multiplicity family,
replicated at three master seeds and on MNIST. The seed-stable,
multiplicity-corrected findings: PBit is nearly indifferent to corruption
noise on Rastrigin where Adam degrades heavily; Adam dominates PBit on
Rosenbrock at both dimensions; on MNIST, PBit trails on clean gradients and is
less sign-robust than Adam, though more robust than SGD. The sign-noise and
dimension-scaling results are direction-consistent but seed-fragile. The
quantization claims are withdrawn as untestable in the cell they came from.
The full registry with numbers is in [`CLAIMS.md`](CLAIMS.md); the rules
behind the numbers are in [`METHOD.md`](METHOD.md); the mechanism is analyzed
in [`THEORY.md`](THEORY.md); the literature positioning is in
[`RELATED.md`](RELATED.md).
