# Why PBit is robust to some gradient noise, and when that robustness vanishes

This note explains the empirical pattern in `CLAIMS.md` and `results/mnist.json`
from the update rule alone, and states what each piece of it does and does not
establish.

## The update, decomposed

For a gradient `g` at inverse temperature `beta`:

```
g_scale = mean(|g|) + eps
prob    = sigmoid(-beta * g / g_scale)     # per coordinate
sigma   ~ Bernoulli(prob) in {+1, -1}      # direction
step    = lr * (|g| + eps) * sigma         # magnitude
```

Two components carry the gradient information: the direction decision `sigma`,
which depends on the gradient only through the normalized value `g / g_scale`,
and the step magnitude, which is proportional to `|g|`.

## Property 1: at high beta, direction is sign-only

As `beta` grows under annealing, `sigmoid(-beta * g / g_scale)` saturates and
`sigma = -sign(g)` with probability approaching one. The *magnitude* of the
normalized gradient stops mattering to the direction decision; only its sign
survives. A monotone transformation of gradient magnitude (scaling,
quantization, amplification) that leaves the sign and the normalization
unchanged leaves the decision distribution nearly unchanged.

This is the structural source of every robustness result in this study. It is
also the source of its main weakness: a sign-only direction discards exactly
the magnitude information a curved valley rewards, which is why Adam outpaces
PBit on Rosenbrock and on MNIST in the clean condition.

## Property 2: under clipping, magnitude is constant

When the gradient magnitude exceeds the clip bound, the clipped gradient has
constant magnitude per saturated coordinate, so PBit's step is `lr * (clip +
eps) * sigma`: magnitude noise is absorbed entirely by the clip. On Rastrigin
at dim=2 the clip fires on every step of every run (measured: 300 of 300 steps,
`results/clip_confound.json`), so in that cell the optimizer sees a
constant-magnitude, sign-only update at high beta.

Two consequences follow:

- The 4-bit quantization comparison (claim C1) is vacuous there: quantization
  preserves endpoints, so clip(quantized) equals clip(raw), and there is
  nothing left for the noise to change.
- The corruption robustness (claim C6) is a property of the optimizer *plus*
  the clip at these gradient scales. Without the clip, corruption would triple
  the step on corrupted components and the trajectory would differ. The
  robustness claim must therefore be stated as "PBit under gradient clipping",
  not "PBit" alone.

## Property 3: sign-flip tolerance is regime-dependent, not structural

Corruption noise flips the sign of a fifth of the components, and clipping
preserves a sign flip (it clips magnitude, not sign). So corrupted coordinates
step in the wrong direction about 20 percent of the time, for both PBit and
Adam. Why does PBit's terminal loss barely move (shift +0.073) while Adam's
rises by 21.6 on Rastrigin?

The answer is the regime, not the direction rule. On Rastrigin the trajectory
is in a wandering regime: the terminal-loss distribution across runs is wide
(sd 11.2 against a mean of 18.3) because runs end scattered across local
minima. Inverting 20 percent of constant-size steps perturbs the wandering
trajectory but not the stationary distribution it samples from, so the
terminal-loss distribution is nearly unchanged.

Adam's failure on the same cell is mechanical: corruption triples 20 percent of
the components, the second-moment estimate `v` inflates nine-fold on those
coordinates, the effective step `lr * m / (sqrt(v) + eps)` collapses on them,
and the optimizer stalls wherever it is.

On MNIST the regime is different: gradients point at structured minima, runs
converge (test loss sd is ~1 percent of the mean), and a wrong-direction step
is a step away from a specific minimum rather than more wandering. There the
same 20 percent of flipped directions cost PBit 10 accuracy points and Adam
only 6.8, because Adam's moment estimates average the flips against the
uncorrupted 80 percent over time. The toy robustness does not transfer, and
Property 3 predicts exactly that: wandering regime, sign flips are cheap;
convergence regime, they are not.

## What this establishes and what it does not

Established by the update rule and the measurements together:

1. PBit's direction is magnitude-insensitive at high beta (Property 1), so
   quantization and magnitude noise cannot hurt it once annealed. Its
   robustness to 1-bit sign-only gradients is the limiting case of the same
   property: it already uses signs.
2. Under clipping, magnitude noise of any size is fully absorbed (Property 2).
3. Sign-flip noise is absorbed only in the wandering regime (Property 3), which
   is why the toy corruption result replicates across seeds and the MNIST
   result reverses it.

Not established:

1. Any claim that PBit is "robust to noise" without the regime and the clip
   being named. The study's positive results are all conditional on both.
2. Any claim about high dimensions or real architectures. The MNIST MLP is one
   small network; nothing here touches the scale at which p-bit hardware
   arguments are usually made.
