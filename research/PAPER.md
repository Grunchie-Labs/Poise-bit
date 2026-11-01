# P-Bit Optimization Under Low-Precision Gradients: An Honest Measurement

> Every number in this paper traces to a recorded artifact in `results/`.
> `research/experiments/verify_docs.py` (run in CI) checks that claim.
> Reproduce everything with the commands in `research/README.md`.

## Abstract

A p-bit is a stochastic binary unit whose state follows a Boltzmann
distribution. Used as an optimizer, it replaces the deterministic gradient
direction with a Bernoulli decision while keeping a gradient-proportional
step magnitude, annealed from exploration toward descent. The natural
question is whether this construction tolerates quantized and noisy
gradients better than standard optimizers. We measure that question under a
protocol designed to be unable to manufacture its own answer: tuned-on-clean
configurations transferred to noise, paired comparisons on shared random
streams, bootstrap intervals with sign-flip tests, Holm correction within each
family (17 main comparisons; 12 in the C8 retest; C9 raw-only and provisional),
seed replication, and a real-task transfer check on MNIST.

Findings: PBit is nearly indifferent to corruption noise on Rastrigin where
Adam degrades by about 21.6 loss units (confirmed, stable at all seeds);
Adam dominates PBit on Rosenbrock at both dimensions (refuted against PBit,
stable); on MNIST PBit trails on clean gradients and is less sign-robust
than Adam but more robust than SGD. The 4-bit quantization comparison is
untestable in its original cell (clipping erases quantization there); this
revision adds a clean retest on an unclipped landscape plus a mechanism
ablation. Eight registered claims across 17 comparisons, one real-task replication,
one 12-contrast quantization retest (C8, provisional), one 6-contrast ablation
(C9, provisional), five figures.

## 1. The update

For gradient `g` at inverse temperature `beta`:

```
g_scale = mean(|g|) + eps
prob    = sigmoid(-beta * g / g_scale)
sigma   ~ Bernoulli(prob) in {+1, -1}
step    = lr * (|g| + eps) * sigma
x      += step
```

Two carriers of gradient information: the direction `sigma`, which depends
on the gradient only through the normalized value `g / g_scale`, and the
step magnitude, proportional to `|g|`. At high `beta` the direction is
sign-only: magnitude stops mattering to the decision, only its sign
survives. That is the structural source of every robustness result here,
and of the main weakness: a sign-only direction discards exactly the
magnitude information a curved valley rewards.

Annealing (`beta(t) = min(beta0 * (1 + t/tau), beta_cap)`) moves the
optimizer from exploration (near-random directions, high entropy) to
exploitation (descent, low entropy).

## 2. Protocol (full rules in METHOD.md)

1. **Tune on clean, transfer to noise.** Declared grids, full scores
   recorded; the selected config moves to noise cells. Per-noise tuning
   turned the noise effect into a selection lottery; one claim flipped
   between the designs, which is how the flaw was found.
2. **Common random numbers.** Init and noise streams shared across
   optimizers; private coins per optimizer identical across noise
   conditions. Every comparison is paired within runs.
3. **Terminal loss, never the min.** The running minimum falls when
   variance rises even if the mean does not (locked by test); claims use
   the terminal value. Whether the readers can disagree is itself claim C5.
4. **Paired statistics, Holm within each family.** Bootstrap CI plus exact sign-flip
   test; a claim resolves only when both agree; the main battery is 17
   comparisons in one Holm family, the C8 retest is a separate 12-contrast
   family on its terminal reader, and C9 reports raw p-values only (TODO: its
   own 6-contrast family).
5. **Replication.** Three master eval seeds; MNIST MLP transfer check
   under the same tune-transfer-evaluate protocol.

## 3. Results (full registry in CLAIMS.md)

- **Corruption on Rastrigin (C6).** PBit shift +0.073, Adam shift +21.559,
  difference -21.486, CI [-28.976, -14.604], Holm p = 9.0e-04, stable 3/3.
  The rock-solid PBit-favorable result: corruption inflates Adam's second
  moment nine-fold on corrupted coordinates while PBit's wandering-regime
  trajectory barely moves (THEORY.md).
- **Rosenbrock (C3, C4).** Adam reaches the valley essentially exactly at
  both dims (dim=2: 1.363 vs 0.0079; dim=10: 7.461 vs 0.971; refuted,
  stable); faster where both succeed (gap 23.5 iters) and never fails
  where PBit fails 13/50.
- **MNIST.** Clean: PBit 93.06% vs Adam 97.25% vs SGD 97.34%. Sign-only:
  PBit 83.09%, Adam 90.47%, SGD 77.89%. Toy ordering reverses on the real
  task; PBit still beats SGD's collapse. Ordering: Adam, PBit, SGD.

Seed-fragile but direction-consistent: Rastrigin dim-scaling (C7),
sign-noise on Rastrigin/Ackley (C2, C6). Withdrawn: all quantized
improvement claims (C1).

## 4. New in this revision (both provisional: seed 0 only)

- **C8 retest on Sphere.** Premise verified in-script: the clip fires on 0 of
  20 probe runs on Sphere dim=10 (`results/quantize_retest.json`) vs 300 of
  300 steps on Rastrigin (`results/clip_confound.json`). Regime: dim=10, 100
  iterations, 50 paired eval runs, tuned-on-clean configs (PBit lr 0.1, tau
  100; Adam lr 0.5; SGD lr 0.5; SignSGD lr 0.05). Twelve contrasts in one Holm
  family on the terminal reader: PBit 4-bit shift -3.85e-18 (Holm 1.0) and
  8-bit shift -3.34e-18 (Holm 0.878) show no detectable difference, with the
  AUC reader agreeing; the PBit 1-bit shift +2.75e-13 (Holm 6.0e-04) sits at
  the solver floor (clean 3.7e-17) while trajectory speed costs AUC +0.141.
  Controls: 1-bit costs SGD +12.756, Adam +0.0691, SignSGD +0.01203. No
  significant terminal shift is negative: quantization never helps. TODO:
  seeds 1/2, Holm over the AUC reader.
- **C9 ablation.** Frozen-beta vs full PBit (isolates annealing);
  deterministic sign-with-PBit-magnitude vs full PBit (isolates noise) — but
  the selected frozen beta0 of 8.0 is saturated, so the two controls twin each
  other (clean 18.327 vs 18.327) and the separation fails (TODO: frozen
  low-beta control). With the tested configs on Rastrigin dim=2, removing
  annealing/noise improves clean terminal (gap -6.712, CI [-8.572, -4.905])
  and corruption-shift robustness (gap -30.909, CI [-38.875, -23.281]), while
  sign noise improves full-PBit terminal (gap +6.237, CI [4.397, 8.134]). The
  ablation PBit's corruption shift (31.258) does not match the main-study
  PBit's (0.073): different selected configs, so C9 does not explain C6
  (TODO: re-run at the main-study configuration; TODO: seeds 1/2, 6-contrast
  Holm family; all C9 p-values are raw).
- **Figures.** Five panels from recorded artifacts only (assets/figs/).

## 5. Limitations

No bare "robust to noise" without regime and clip; nothing beyond one
small MLP at real scale; selected configs are "one of several equivalent"
on flat landscapes; threshold claims carry failure counts (13/50 vs 0/50).

