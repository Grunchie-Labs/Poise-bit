# Measurement method

This file states the rules the numbers in `CLAIMS.md` were produced under. Each
rule exists because dropping it changed a conclusion, and several were added
only after a claim was watched flipping when the rule was removed.

## Common random numbers

The runner draws randomness from three SHA256-derived streams per run, keyed on
different parts of the configuration:

- `init`: keyed on (function, run). Every optimizer and every noise condition
  shares one start position.
- `noise`: keyed on (function, noise, run). Every optimizer within one noise
  condition sees the same gradient perturbation.
- `opt`: keyed on (function, run, optimizer). A stochastic optimizer's private
  coins are identical across noise conditions but independent across
  optimizers.

Because run `i` of optimizer A and run `i` of optimizer B share a start and a
perturbation, their terminal losses are paired observations, and differencing
within a run cancels that shared variance. On these landscapes the seed-to-seed
standard deviation of terminal loss is often comparable to the mean, so an
unpaired comparison needs far more runs to see the same effect. The unit test
`test_pairing_beats_unpaired_on_shared_start_variance` asserts the paired
standard deviation is at least five times smaller than the unpaired one on data
with the same shared structure.

Two stream bugs were found and fixed during this study, both verified by
runtime checks rather than by reading:

1. The optimizer name was included in the init and noise streams, so
   optimizers in one cell did not share starts. The module docstring claimed
   they did, and the test that was supposed to check it asserted nothing.
2. The noise name was included in the init and opt streams, so clean and noisy
   runs of one optimizer started at different points with different private
   coins, making cross-noise pairing impossible.

Regression tests now assert all three guarantees directly:
`test_common_random_numbers_pair_optimizers_within_a_cell`,
`test_start_position_shared_across_noise_conditions`, and
`test_private_rng_stream_identical_across_noise_conditions` in
`tests/test_experiment.py`.

## Equal objective budget

Every optimizer gets one objective evaluation per iteration. The gradient path
evaluates once after stepping; the ask/tell path evaluates once per candidate.

This rule is enforced by
`test_every_optimizer_family_gets_the_same_objective_budget`, which counts
evaluations for SGD, PBit, simulated annealing, and an evolution strategy and
asserts they all match. Before that test existed, the ask/tell branch evaluated
each candidate once in the runner and again inside `tell()`, so ask/tell
optimizers consumed twice the budget of every other method. The prior results
table was produced under that condition.

Ask/tell optimizers are still not compared at equal iteration count with
gradient methods in this study (an evolution strategy with `lam=10` performs
one center update per ten evaluations), and the ask/tell path does not apply
gradient noise at all, since zeroth-order methods do not consume gradients.
Their rows from the prior table are treated as unmeasured.

## Terminal loss, not the minimum

Two statistics are recorded for every run:

- `final_current`: the loss at the last iteration.
- `final_best`: the best loss seen anywhere in the trajectory.

The second is a running minimum, so it falls when trajectory variance rises
even when the mean loss does not.
`tests/test_metrics.py::test_min_over_trajectory_is_variance_sensitive` holds
the mean fixed, adds variance, and asserts the minimum moves at least five
times as much as the mean. All claims here are computed on the terminal value;
the minimum is recorded for inspection and used by claim C5, which checks
whether the two readings can disagree.

## Tuning protocol

Hand-picked hyperparameters favour whichever method the author happened to tune
for, and Adam and SGD have very different scale requirements, so every
optimizer in this study is tuned per function and dimension:

1. A declared grid per optimizer (`results/tuned.json` records every grid
   point and its score, so the selection can be audited rather than trusted).
2. Selection on 10 runs from a tuning seed, minimizing terminal current loss.
3. Evaluation on 50 runs from a different seed. The two seeds are required to
   differ; `assert_seeds_disjoint` raises otherwise. Selecting and reporting on
   the same seeds would leak the search into the result.

Two further rules were learned the hard way:

**Tune on clean, transfer to noise.** An earlier version of this study tuned
each optimizer inside each noise cell. Where the tuning landscape is flat (on
Rastrigin the top PBit configurations scored 19.9987, 20.0090, and 20.0286),
the per-cell selections are effectively a lottery, and a "noise effect"
computed from two different configurations measures the lottery. One claim
(C2) flipped from confirmed to unresolved between a 5-run and a 10-run tuning
budget for exactly this reason. With transfer, clean and noisy runs share one
configuration and the shift isolates the noise.

**Flat landscapes make selection itself uncertain.** Even on clean cells the
top of the Rastrigin grid is nearly level, so the selected configuration should
be read as "one of several equivalent" rather than "the best". The grid record
shows the runner-up scores; where they are within noise of the winner, the
choice is arbitrary and the eval-phase pairing, not the selection, is doing the
work.

## Uncertainty

Every mean is reported with a standard deviation and a 95 percent percentile
bootstrap interval. Comparisons between conditions use a paired test:

- The interval comes from resampling the within-run differences.
- The p-value comes from a sign-flip randomization test, exact under the null
  that the sign of a within-run difference is equally likely either way. The
  per-run loss differences on these landscapes are heavy-tailed, which is why
  a t-test is not used.
- A claim resolves only when the interval and the test agree. When they
  disagree the claim is recorded as inconsistent, since a conclusion that
  depends on which statistic is consulted is not a conclusion. This fired once
  during development (at n=3) and the verdict logic was changed because of it.

All claims form one family and get a Holm correction, reported as
`p_value_holm`. A scale-aware zero floor is applied before reading the sign of
any difference: with a true effect of zero the two readers in C5 returned mean
differences of about 1e-16 with opposite signs, and an earlier version reported
that numerical noise as a substantive disagreement.

## Power analysis

`experiments/power.py` subsamples the recorded 50-run series without re-running
any optimization. For the confirmed sign-noise effect it reports the detection
rate at smaller sample sizes; for the exactly-zero quantization comparison it
reports the false-positive rate. The point is that a study at n=5 (the prior
report's size) detects the real effect 0.0 percent of the time, while the null
is called significant 0.0 percent of the time at every size.

## Seed replication

Every claim is measured at three master eval seeds (0, 1, 2) with tuning
unchanged, because a verdict that depends on the master seed is a verdict about
noise. `experiments/robustness.py` compares the battery across seeds on two
axes:

- **Category stability**: does the claim land in the same verdict category
  (resolved direction, unresolved, inconsistent) at every seed?
- **Direction stability**: does the sign of the effect ever flip? A
  scale-aware floor (1e-9) keeps machine-epsilon diffs from counting as flips,
  the same lesson as the C5 epsilon bug.

Current state: 13 of 20 categories stable, 17 of 20 directions stable. The
three direction-flipping rows are all near-zero effects that never resolve.
This distinction matters for how the claims are worded: direction-stable but
category-fragile claims are reported as "direction consistent, magnitude
seed-dependent", which is exactly what C2 and C7 are.

## Real-task replication (MNIST)

Toy landscapes established the mechanism; a real task tests whether it
transfers. `experiments/mnist.py` trains an MLP (layer sizes 784, 128, 10,
ReLU) on MNIST under the same protocol as the toy study:

1. Tune each optimizer's learning rate on a held-out validation split (the
   last 10000 training examples) under clean gradients, 3 runs per grid point.
2. Transfer the selected configuration to the sign-only gradient condition,
   mirroring the clean-tune-transfer rule.
3. Evaluate on the test set over 15 paired runs of 10 epochs. Run `s` gives
   every optimizer the same weight init and the same data order, so the
   comparisons are paired exactly as in the toy study. PBit's Bernoulli
   sampling is seeded per run.

The sign condition replaces each gradient with its component-wise sign after
backpropagation, the same transform as the toy study's SignNoise.

The MNIST study is deliberately small: one architecture, one dataset, 10
epochs. It is a transfer check on the headline claims, not a bid for
state-of-the-art accuracy.

## Known limitations

**The quantization cell is confounded by clipping.** Rastrigin gradients at
dim=2 reach component magnitudes above 71, and the runner clips gradients to
5.0, which fires on every step of every run in that cell (300 of 300 steps in
all 20 sampled runs). Quantization preserves the raw endpoints, so the clipped
vector is identical whether or not the gradient was quantized first (1000 of
1000 random large gradients checked; see `results/clip_confound.json`). A
quantization claim needs a landscape whose gradients fit the clip range, or a
larger clip.

**PBit can coincide with SGD.** At moderate inverse temperature with
proportional steps, PBit picks the descent direction with high probability per
coordinate, and over a short trajectory its path can be bit-identical to SGD at
the same learning rate. Where the two agree, a "PBit versus SGD" comparison
measures nothing; the leaderboard shows several cells where their means match
to three decimals. This is the annealing story working as designed (the update
becomes gradient descent as it cools), but it means PBit's distinctness lives
in the high-temperature phase and in its insensitivity to magnitude noise.

**AdamW selected weight_decay = 0.0 everywhere**, which makes it identical to
Adam. Its grid included nonzero decay and the record shows decay never won on
these landscapes, so the two rows agreeing is a result, not a bug.

**Three toy functions, two dimensions.** Rastrigin, Ackley, and Rosenbrock at
dim=2, plus clean cells at dim=10. Nothing here addresses real loss surfaces,
and the dim=10 cells are clean only.

**Tuning grids are small** (a handful to fifteen points per optimizer) and the
tuning criterion is terminal loss at 300 iterations. A configuration that wins
at 300 iterations may not be the one that wins at a longer budget.

**Threshold comparisons drop failures.** Claim C4 compares only runs where both
optimizers reached the threshold and reports the dropped runs as counts
(13 of 50 for PBit, 0 of 50 for Adam). Comparing successes alone would flatter
whichever method fails more often, so the failure counts are part of the
result, not a footnote to it.
