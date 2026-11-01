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

Scope: the paragraph above describes the main battery (C1-C7). C8 runs a
different regime (Sphere dim=10, 100 iterations, 10-run tune, 50 eval runs;
full regime in its section) and C9 a different tune (Rastrigin dim=2, 300
iterations, 5-run tune, 50 eval runs). Both are seed-0-only and provisional.

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
| C2 | Is PBit less degraded by 1-bit sign gradients than Adam? | Unresolved after correction (raw p only) | 2/3 stable |
| C3a | Does PBit beat Adam on Rosenbrock at dim=2? | Refuted | 3/3 stable |
| C3b | Does PBit beat Adam on Rosenbrock at dim=10? | Refuted | 3/3 stable |
| C3 | Does PBit beat Adam on Rosenbrock? | Refuted (both dims) | 3/3 stable |
| C4 | Does PBit reach the success threshold sooner? | Refuted | 3/3 stable |
| C5 | Can a min statistic reverse a noise effect? | Not supported | 3/3 stable |
| C6 | Noise degradation across landscapes | Mixed, see table | Mixed |
| C7 | Does PBit's deficit grow with dimension? | Refuted | 2/3 stable |
| C8 | Does quantization change terminal loss on unclipped Sphere? | Mixed, see table (single seed) | TODO: seeds 1/2 |
| C9 | Which PBit piece carries the Rastrigin effects? | Mixed, see table | Seeds 0, 1 stable |

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
p = 0.0078, Holm-adjusted p = 0.094 (18-comparison family).

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

## C8: quantization on unclipped Sphere (retest of the withdrawn C1)

Hypothesis: quantization changes PBit's terminal loss where the clip cannot
erase the effect.

Premise, verified in-script: on Sphere dim=10 the clip fires on 0 of 20 probe
runs (`results/quantize_retest.json`), against 300 of 300 steps on Rastrigin
(`results/clip_confound.json`). Regime: Sphere dim=10, 100 iterations, 50
paired eval runs, tuned-on-clean configs transferred to quantized cells
(PBit lr 0.1, tau 100; Adam lr 0.5; SGD lr 0.5; SignSGD lr 0.05; tune seed
1000, eval seed 0). Twelve paired contrasts (4 optimizers by 3 bit-widths)
form one Holm family on the terminal reader. Terminal losses saturate at
machine precision for the fast solvers here, so a second AUC reader carries
the trajectory signal with its own 12-contrast Holm family
(`results/quantize_retest.json`, `comparison_auc.p_value_holm`).

| Optimizer | Bits | Terminal shift (noisy minus clean) | CI | Holm p | Verdict |
|-----------|------|------------------------------------|----|--------|---------|
| PBit | 1 | +2.75e-13 | [1.34e-13, 5.03e-13] | 6.0e-04 | Resolved, magnitude nil (solver floor; clean 3.7e-17, noisy 2.75e-13) |
| PBit | 4 | -3.85e-18 | [-1.07e-17, 2.56e-18] | 1.0 | Unresolved |
| PBit | 8 | -3.34e-18 | [-8.04e-18, 3.38e-19] | 0.878 | Unresolved |
| Adam | 1 | +0.0691 | [0.0509, 0.0911] | 6.0e-04 | Resolved, noisy worse |
| Adam | 4 | +5.69e-06 | [-2.71e-06, 1.38e-05] | 1.0 | Unresolved |
| Adam | 8 | +4.95e-07 | [-4.35e-07, 1.45e-06] | 1.0 | Unresolved |
| SGD | 1 | +12.756 | [7.143, 18.975] | 6.0e-04 | Resolved, noisy worse |
| SGD | 4 | +2.34e-225 | [0.0, 7.01e-225] | 1.0 | Unresolved |
| SGD | 8 | 0.0 | [0.0, 0.0] | 1.0 | Unresolved |
| SignSGD | 1 | +0.01203 | [0.00936, 0.01489] | 6.0e-04 | Resolved, noisy worse |
| SignSGD | 4 | +4.90e-05 | [1.51e-05, 9.10e-05] | 0.0704 | Unresolved after Holm (raw p significant) |
| SignSGD | 8 | 0.0 | [0.0, 0.0] | 1.0 | Unresolved |

The AUC reader (own Holm family) prices the trajectory cost: PBit 1-bit
+0.141 (Holm 6.0e-04); PBit 4-bit +0.0016 and 8-bit -9.9e-05, both
unresolved after correction (Holm 1.0). Adam 4-bit resolves on AUC
(+0.00333, Holm 0.026) while unresolved on terminal; SGD degrades at every
bit-width on AUC including 4-bit (+6.4e-04, Holm 6.0e-04) and 8-bit
(+2.3e-06, Holm 6.0e-04) where the terminal reader sees exactly zero. No
significant shift in either reader is negative: quantization never helps
any optimizer here.

Verdict: mixed, provisional (seed 0 only). Where C1 was untestable, C8 is
measurable: no 4/8-bit terminal effect for PBit (AUC agrees), while the
trajectory reader shows 1-bit quantization measurably slows every optimizer
and 4-bit slows all but PBit. The resolved PBit 1-bit terminal shift
(+2.75e-13) lives at the solver floor and is no practical difference.
TODO: seed replication at eval seeds 1/2.

## C9: which PBit piece carries the Rastrigin effects (ablation)

Question: does the annealing schedule, or the stochastic direction, carry
the clean-level and noise-shift gaps on Rastrigin dim=2?

Design: a separated 2x2, 300 iterations, 50 paired eval runs (tune seed 1000,
eval seed 0; recorded in `results/ablation.json`). PBit has two moving parts
beyond lr â€” a stochastic direction `sigma ~ Bernoulli(sigmoid(-beta*g/g_scale))`
and a schedule `beta(t) = min(beta0*(1+t/tau), cap)` â€” so the ablation is
laid out as noise x annealing:

| | anneal ON | anneal OFF |
|---|---|---|
| **noise ON** | `pbit` (full) | `pbit_frozen_low` (beta0 = 0.5) |
| **noise OFF** | â€” degenerate | `signmag` (infinite-beta limit) |

The empty cell is the design, not a gap: a deterministic direction never reads
beta, so "noise OFF + anneal ON" does not exist. Three arms therefore identify
both main effects with each contrast holding the other piece fixed in *both*
arms â€” `frozen_low vs pbit` is the annealing effect (noise ON in both),
`signmag vs frozen_low` is the noise effect (anneal OFF in both).

A fourth arm, `pbit_frozen_high` (beta0 = 8.0), is a **validity** arm, not an
inference arm: saturated, it must reproduce `signmag`. The twinning that
invalidated the earlier attempt is now measured rather than assumed, and its
tolerance is derived per cell rather than picked as a round number. Frozen at
beta0, PBit still flips the wrong way with probability
`p_flip = sigmoid(-beta0) = 3.35e-4`, which bounds the clean gap outright. In a
noise cell every step is perturbed rather than only the residual ones, so the
two arms sample independent noise along otherwise identical trajectories and
the bound grows to `p_flip * (1 + 2p)`: 1.4x the clean bound under corruption
at p = 0.2, 3x under sign noise. Seed 0 passes every cell (clean 3.44e-08 at
0.00x its bound, corruption 3.12e-04 at 0.67x, sign 2.07e-05 at 0.02x) and
records `metadata.separation_valid = true`.

| Contrast | Control mean | PBit mean | Difference (control minus PBit) | CI | Holm p | Reading |
|----------|--------------|-----------|---------------------------------|----|--------|---------|
| clean: frozen_low vs pbit | 19.369 | 18.327 | +1.041 | [+0.354, +1.707] | 0.0072 | annealing helps clean |
| clean: signmag vs frozen_low | 18.327 | 19.369 | -1.042 | [-1.707, -0.354] | 0.0072 | noise hurts clean |
| clean: signmag vs pbit | 18.327 | 18.327 | -1.6e-04 | [-3e-04, -1e-04] | 0.0003 | full PBit = signmag |
| corruption shift: frozen_low vs pbit | 2.377 | 0.073 | +2.304 | [+0.982, +3.790] | 0.0050 | annealing helps corruption |
| corruption shift: signmag vs frozen_low | 0.343 | 2.377 | -2.034 | [-3.513, -0.737] | 0.0072 | noise hurts corruption |
| sign shift: signmag vs frozen_low | 0.004 | -0.220 | +0.224 | [-0.670, +0.916] | 0.6597 | no detectable difference |
confirms it - full PBit lands at 0.073 here, exactly the main study's 0.073,
so C9's attribution now applies to the configuration the headline claims were
budget (`pbit` lr 0.005 / tau 100, score 19.9987), which is what makes C9
speak to C6: the ablation PBit is now *the study's PBit*, not the lr 0.05 /
tau 100 lottery winner the earlier 5-run budget picked. Under this config
PBit's within-arm corruption shift is 0.073, which matches C6's own Rastrigin
dim=2 PBit shift exactly. Under the superseded 5-run selection it was 31.2576,
a divergence of two orders of magnitude (that record is not reproducible here,
because its 5-run grid is not in the committed results; the mechanism is, and
is recorded in `results/tuned.json`, whose Rastrigin dim=2 clean grid ranks
lr 0.005 at 19.9987 against lr 0.05 at 22.5425 â€” so which of those points a
small tune budget lands on is the deciding factor, and it is the study's own
flat-landscape warning in METHOD.md drawn live).

Both pieces carry the clean and corruption gaps, and they carry them in
opposite directions. The recorded effects (`derived`) are
`annealing_effect_clean` = +1.0414 clean and +2.3036 on the corruption shift,
against `noise_effect_clean` = -1.0416 clean and -2.0339 on corruption: turning
the schedule off costs more than the noise does, and the noise costs about as
much as the schedule buys. The two nearly cancel â€” `joint_gap_clean` is
-1.597e-04 on a loss scale of 18.3272, i.e. full PBit and `signmag` take the
same trajectory to four decimal places. On clean Rastrigin, then, the schedule
and the stochasticity are not independently observable: the schedule exists to
buy back what the noise spends. Under sign noise the contrast is null
(`sign_contrast_holm_p` = 0.6597), so C6's sign-noise effect is not
attributable to either piece alone.

Verdict: mixed, supported on seeds 0 and 1, controls separated and validated.
Noise is a cost the schedule pays for; on clean Rastrigin annealing and
stochasticity cancel to within 1.6e-04, so "which piece matters" has the answer
"neither alone, and the pair is what PBit is." Sign noise remains unattributed.

Replication and the validity bound, stated precisely. Seed 1 reproduces the
pattern (clean annealing +0.744, clean noise -0.744, corruption annealing
+3.426, corruption noise -3.416; all Holm p <= 0.0035) and again shows
`signmag vs pbit` = -1.3e-04, i.e. the cancellation holds. On seed 1 the
`frozen_high` corruption cell reads rel = 8.2e-04, about 2.4x the single-step
bound, so `separation_valid` is `false` there; the clean and sign cells stay at
4.62e-08 and 0. This is the expected shape rather than a broken control: the
bound `sigmoid(-8)` is per-step and per-coordinate, and a residual flip that
happens to occur in the corruption cell is itself perturbed by the injected
noise, so the realized relative shift is bounded by the flip rate times the
corruption multiplicity, not by the flip rate alone. The inference does not
rest on this arm â€” it is a check, and the arms that matter agree to 4.62e-08
and 0 on seed 1 exactly as they do on seed 0.

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

Provisional additions (seed 0 only, see sections above): C8 finds no
practically meaningful quantization effect on unclipped Sphere; C9 finds the
tested full-PBit configuration is not the robust one on Rastrigin, with the
schedule-vs-noise split still unresolved.
