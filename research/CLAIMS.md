# Claims and verdicts

Every number below is produced by scripts in `experiments/`, recorded in
`results/`, and checked against this prose by `experiments/verify_docs.py`,
which also runs in the test suite.

Measurement conditions: each optimizer tuned per function and dimension on 10
clean runs from a tuning seed, evaluated on 50 held-out runs from a different
seed; 300 iterations per run; Rastrigin, Ackley, and Rosenbrock at dim=2, plus
clean cells at dim=10. The tuning criterion is terminal current loss. The full
grid and every score are in `results/tuned.json`, and the selected
configuration of every optimizer is in `results/leaderboard.csv`.

Two replication layers sit on top of the base measurement:

- **Seed replication.** The whole battery is re-run at three master eval seeds
  (0, 1, 2) with tuning unchanged. A verdict is stable when it lands in the
  same category at every seed; an effect direction is stable when the sign of
  the difference never flips. Results: 13 of 20 verdict categories stable, 17
  of 20 directions stable (`results/robustness.json`). The three
  direction-flipping rows are all near-zero effects that never resolve; every
  effect that reaches significance at any seed keeps its direction.
- **Real-task replication.** The headline comparisons are re-run on MNIST with
  a small MLP (`results/mnist.json`), under the same tune-transfer-evaluate
  protocol with paired runs.

Two protocol changes invalidate every earlier version of this table, including
the pre-registration draft that existed during development:

1. The common-random-number streams were repaired. Start positions are now
   shared across optimizers and across noise conditions within a run, and a
   stochastic optimizer's private coins are shared across noise conditions.
   Before this fix, comparisons that were described as paired were not paired.
2. Tuning now happens on clean cells only and the selected configuration is
   transferred to noise cells. The earlier per-cell design let each optimizer
   adapt to each noise condition, and on flat tuning landscapes the selections
   were effectively a lottery: a "noise effect" computed from two different
   configurations measures the lottery, not the noise. One claim flipped
   between the two designs, which is how the flaw was found.

## Verdict summary

| ID | Question | Verdict (seed 0) | Seed stability |
|----|----------|------------------|----------------|
| C1 | Does 4-bit quantization change PBit's terminal loss? | Unresolved, exactly zero | 3/3 stable |
| C2 | Is PBit less degraded by 1-bit gradients than Adam? | Resolved (raw p only) | 2/3 stable |
| C3a | Does PBit beat Adam on Rosenbrock at dim=2? | Refuted | 3/3 stable |
| C3b | Does PBit beat Adam on Rosenbrock at dim=10? | Refuted | 3/3 stable |
| C4 | Does PBit reach the success threshold sooner? | Refuted | 3/3 stable |
| C5 | Can a min statistic reverse a noise effect? | Not supported | 3/3 stable |
| C6 | Noise degradation across landscapes | Mixed, see table | Mixed |
| C7 | Does PBit's deficit grow with dimension? | Refuted | 2/3 stable |

## C1: effect of 4-bit stochastic quantization on PBit

Hypothesis: noise cannot make the terminal loss lower than clean.

Paired difference (quantized minus clean), terminal current loss, Rastrigin
dim=2: mean difference 0.0, sd 0.0, p = 1.0. Clean mean 18.327, sd 11.242.

Verdict: unresolved, and the reason is a confound, not robustness. Rastrigin
gradient components reach magnitudes above 71, and the runner's gradient clip
at 5.0 fires on every step of every run (300 of 300 steps in all 20 sampled
runs). Quantization maps components to 16 levels between their own minimum and
maximum, preserving the endpoints, so the clipped vector is identical whether
or not the gradient was quantized first (1000 of 1000 random large gradients
checked; `results/clip_confound.json`). After clipping there is nothing left
for 4-bit quantization to change.

So this cell cannot detect a quantization effect at all. The prior report's
claim that PBit improved under 4-bit quantization (11.69 against 23.94 clean)
fails on three independent grounds: it does not reproduce at 50 paired runs,
the resampling in `results/power.json` shows what pure noise looks like at
that sample size, and the comparison itself is confounded by clipping.

## C2: degradation under 1-bit sign-only gradients

Hypothesis: PBit shifts less than Adam.

PBit shift: mean 0.004, sd 0.002. Adam shift: mean 1.324, sd 3.480. Paired
difference of shifts: mean -1.320, sd 3.480, 95 percent CI [-2.320, -0.417],
p = 0.0078, Holm-adjusted p = 0.094 (17-comparison family).

Verdict: weak evidence only. The raw test is significant, the corrected one is
not, and the verdict is seed-fragile (resolved at seeds 0 and 1, unresolved at
seed 2; the direction is negative at all three). The prior report's version of
this claim survives only in the weakened form "the direction of the effect is
consistent, its magnitude is small, and the study is underpowered to confirm
it after multiplicity correction."

The power analysis subsamples the recorded series (`results/power.json`): at
n=5 the effect is detected 0.0 percent of the time, at n=10 18.5 percent, at
n=20 33.1 percent, at n=30 55.4 percent, at n=50 100 percent. A study at the
prior report's sample size had essentially no chance of finding it.

## C3: terminal loss on Rosenbrock

Hypothesis: PBit reaches lower terminal loss than Adam at equal budget.

dim=2: PBit 1.363 (sd 2.048), Adam 0.0079 (sd 0.038). Paired difference 1.355,
CI [0.820, 1.943], p = 5.0e-05, Holm-adjusted p = 9.0e-04.

dim=10: PBit 7.461 (sd 1.502), Adam 0.971 (sd 1.719). Paired difference 6.489,
CI [6.040, 6.925], p = 5.0e-05, Holm-adjusted p = 9.0e-04.

Verdict: refuted at both dimensions, stable at all three seeds. Adam reaches
the Rosenbrock valley essentially exactly; PBit does not. Tuning helped PBit
considerably (its untuned deficit was far larger), and it did not change the
ordering.

## C4: iterations to threshold on Rosenbrock

Hypothesis: PBit needs fewer iterations to reach loss below 1.0.

Compared on the 37 of 50 runs where both optimizers succeeded. PBit mean hit
time 48.8 iterations, Adam 25.3. Paired difference 23.5, CI [18.568, 28.676],
p = 5.0e-05, Holm-adjusted p = 9.0e-04.

Failure counts: PBit failed to reach the threshold in 13 of 50 runs. Adam
failed in 0 of 50.

Verdict: refuted, stable at all three seeds. Adam is faster where both
succeed, and PBit fails on about a quarter of runs where Adam never fails.

## C5: whether a min-over-trajectory statistic reverses the sign

On the C1 comparison both readers report an effect of exactly zero, so they
agree. Verdict: not supported here. The mechanism is real and locked by
`tests/test_metrics.py::test_min_over_trajectory_is_variance_sensitive`, but
the data does not show it reversing any comparison in this study. A
development version of this check reported a spurious disagreement from a
1e-16 sign flip between two zero effects, which is why the comparison now
applies a scale-aware zero floor before reading signs.

## C6: degradation across noise families and landscapes

Same construction as C2, per (landscape, noise), dim=2, tuned-on-clean
configurations. PBit shift and Adam shift are the noisy-minus-clean change in
terminal current loss; the comparison is their paired difference.

| Landscape | Noise | PBit shift | Adam shift | Difference | CI | Holm p | Verdict |
|-----------|-------|-----------|------------|------------|----|--------|---------|
| Rastrigin | Gaussian | 0.022 | 0.076 | -0.053 | [-0.722, 0.626] | 1.0 | Unresolved |
| Rastrigin | Corruption | 0.073 | 21.559 | -21.486 | [-28.976, -14.604] | 9.0e-04 | Confirmed, 3/3 stable |
| Rastrigin | Quantize 4-bit | 0.000 | 0.008 | -0.008 | [-0.156, 0.138] | 1.0 | Unresolved (confounded, C1) |
| Rastrigin | Sign (1-bit) | 0.004 | 1.324 | -1.320 | [-2.320, -0.417] | 0.094 | Weak, 2/3 stable |
| Ackley | Gaussian | -0.033 | 0.157 | -0.190 | [-0.816, 0.324] | 1.0 | Unresolved |
| Ackley | Corruption | 0.027 | -0.013 | 0.040 | [-0.981, 1.340] | 1.0 | Unresolved |
| Ackley | Quantize 4-bit | -0.029 | 0.000 | -0.029 | [-0.299, 0.243] | 1.0 | Unresolved |
| Ackley | Sign (1-bit) | -0.483 | 0.270 | -0.753 | [-1.182, -0.217] | 0.032 | Confirmed, 2/3 stable |
| Rosenbrock | Gaussian | -0.099 | 0.001 | -0.100 | [-0.221, -0.007] | 0.544 | Weak, 3/3 unstable |
| Rosenbrock | Corruption | 0.213 | 0.771 | -0.558 | [-1.009, -0.166] | 0.103 | Weak, 2/3 stable |
| Rosenbrock | Quantize 4-bit | -0.000 | 0.000 | -0.000 | [-0.000, 0.000] | 1.0 | Unresolved |
| Rosenbrock | Sign (1-bit) | 6.298 | 0.518 | 5.780 | [1.084, 12.801] | 0.126 | PBit worse, 3/3 stable |

Three things to read in this table:

1. The one rock-solid PBit-favorable result in the study: corruption noise on
   Rastrigin costs Adam about 21.6 loss units and costs PBit 0.07, significant
   after correction and stable at every seed. The mechanism is analyzed in
   `THEORY.md`: corruption inflates Adam's second moment nine-fold on the
   corrupted coordinates, while PBit's wandering-regime trajectory is nearly
   unchanged by 20 percent wrong directions.
2. The sign-noise story is landscape-dependent. PBit is less degraded on
   Rastrigin and Ackley, but *more* degraded on Rosenbrock, stably: on a
   curved valley, sign-only gradients strip the magnitude information PBit's
   annealed updates no longer carry, and Adam's moment estimates average the
   flips out better over time.
3. Ackley is a null landscape for this study: nothing separates the two
   optimizers there under any noise condition.

## C7: dimension scaling of the PBit-Adam gap on Rastrigin

Hypothesis: PBit's deficit against Adam grows from dim=2 to dim=10.

The gap (PBit minus Adam terminal loss) is -1.03 at dim=2 and -6.06 at dim=10,
paired difference 5.03, CI [2.467, 7.475], p = 3.5e-04, Holm-adjusted
p = 0.0049.

Verdict: refuted, but seed-fragile. The direction is positive at all three
seeds (+5.03, +1.39, +2.91) and resolved at two of three. On Rastrigin, PBit
is ahead of Adam at both dimensions and the advantage grows with dimension,
but the magnitude varies enough across seeds that the study cannot pin it
down. The prior report's "Adam dominates in higher dimensions" is not
supported on this landscape; neither, at full strength, is the reverse.

## Real-task replication: MNIST

The headline comparisons re-run with an MLP (layer sizes 784, 128, 10, ReLU)
on MNIST,
15 paired runs of 10 epochs, tuned on a validation split and transferred, per
`results/mnist.json`.

Clean condition: PBit 93.06 percent test accuracy (loss 0.248), Adam 97.25
(loss 0.127), SGD 97.34 (loss 0.086). PBit trails Adam by 4.19 accuracy
points (CI [-0.043, -0.041], p = 5.0e-05) and SGD by 4.28 points
(CI [-0.044, -0.042], p = 5.0e-05).

Sign-only gradients: PBit drops to 83.09 percent (loss 20.48), Adam to 90.47
(loss 5.30), SGD to 77.89 (loss 14.46). Paired on the degradation:

- Loss shift: PBit degrades 15.06 more than Adam (CI [14.203, 16.033],
  p = 5.0e-05) and 5.85 more than SGD (CI [3.408, 8.252], p = 5.0e-04).
- Accuracy shift: PBit degrades 3.19 points more than Adam (CI [-0.045,
  -0.019], p = 5.0e-04) and 9.48 points less than SGD (CI [0.058, 0.137],
  p = 2.0e-04).

The toy-task ordering (PBit more sign-robust than Adam) reverses on the real
task: Adam absorbs sign noise better than PBit where gradients point at
structured minima. The toy robustness was regime-dependent, exactly as
`THEORY.md` predicts: sign flips are cheap in a wandering multimodal regime
and costly in a convergence regime. PBit still beats SGD's collapse under
sign noise, so the honest ordering on this task is Adam, then PBit, then SGD.

## What survives

Seed-stable and multiplicity-corrected:

1. PBit is nearly indifferent to corruption noise on Rastrigin where Adam
   degrades by about 21.6 loss units (C6).
2. Adam dominates PBit on Rosenbrock on terminal loss and threshold speed, at
   both dimensions tested (C3, C4).
3. On MNIST, PBit is behind on clean gradients and less sign-robust than Adam
   (but more robust than SGD).

Direction-consistent but seed-fragile or correction-failing:

4. PBit leads Adam on Rastrigin with the advantage growing from dim=2 to
   dim=10 (C7).
5. PBit is less degraded than Adam under sign-only gradients on Rastrigin and
   Ackley (C2, C6), and more degraded on Rosenbrock (C6, stable).

Withdrawn:

6. Every claim of improvement under quantization (C1): unreproducible at
   paired n=50, underpowered at the original n=5, and untestable in the cell
   it came from because clipping saturates the gradients there.

The pattern across the whole study: PBit's advantages are real but
conditional, on the landscape, the regime, and the clip. Its disadvantages on
smooth valleys and real tasks are not conditional. That is the opposite of
the story the withdrawn table told, and it is the story the measurement
supports.
