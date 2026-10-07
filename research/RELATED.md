# Related work and novelty positioning

This file maps the study onto the literature it sits in. Every citation here was
verified by fetching the paper or its official record during this project;
nothing is cited from memory.

## P-bits and probabilistic computing

The p-bit concept and the update rule this library uses are established work:

- **Camsari, Faria, Sutton, Datta, "Stochastic p-bits for Invertible Logic",
  Phys. Rev. X 7, 031014 (2017).** DOI: 10.1103/PhysRevX.7.031014. Introduces
  the p-bit as an unstable stochastic unit and identifies it with the binary
  stochastic neuron: a sigmoid activation over a weighted input, sampled into
  a binary state. The optimizer's flip probability
  `sigmoid(-beta * g / g_scale)` is this rule with the gradient in the input
  role and annealing in the temperature role.
- **Camsari, Sutton, Datta, "p-bits for probabilistic spin logic",
  Applied Physics Reviews 6, 011305 (2019).** DOI: 10.1063/1.5055860. The
  review article; positions p-bits as a hardware accelerator for binary
  stochastic neurons and for Ising-style optimization.
- **Sutton, Camsari, Behin-Aein, Datta, "Intrinsic optimization using
  stochastic nanomagnets", Scientific Reports 7, 44370 (2017).** p-bit
  networks performing combinatorial optimization.
- **Borders et al., "Integer factorization using stochastic magnetic tunnel
  junctions", Nature 573, 390-393 (2019).** DOI: 10.1038/s41586-019-1557-9.
  Hardware p-bit network solving factorization.
- **Aadit et al., "Massively parallel probabilistic computing with sparse
  Ising machines", Nature Electronics 5, 460-468 (2022).** DOI:
  10.1038/s41928-022-00774-2.
- **Kaiser et al., "Hardware-aware in situ learning based on stochastic
  magnetic tunnel junctions", Physical Review Applied 17, 014016 (2022).**
- **Niazi et al., "Training deep Boltzmann networks with sparse Ising
  machines", Nature Electronics 7, 610-619 (2024).**
- **Chowdhury et al., "A full-stack view of probabilistic computing with
  p-bits", IEEE J. Explor. Solid-State Comput. Devices Circuits 9, 1 (2023).**

What this body of work does: p-bits as physical substrates for invertible
logic, Ising machines for combinatorial optimization, simulated annealing
engines, and Boltzmann-machine learning. What it does not do, as far as this
project's searches found: use the p-bit/BSN update as an iterative optimizer
for *continuous* gradient-descent problems benchmarked against Adam, SGD, and
their variants on standard test landscapes and real tasks. That usage is this
repository's subject, and no prior instance of it was found.

## Sign-based and low-precision gradient methods

- **Bernstein, Wang, Azizzadenesheli, Anandkumar, "signSGD: Compressed
  Optimisation for Non-Convex Problems", ICML 2018, PMLR 80:560-569.**
  arXiv:1802.04434. Shows 1-bit sign-only gradients can match SGD's
  convergence rate, and matches Adam on deep Imagenet models with the momentum
  variant. This is the closest ML-side result to the sign-noise claims here,
  and it is the right framing for them: 1-bit gradients carrying enough
  information is established; our C2 and C6 rows are consistent with it, and
  the MNIST reversal (Adam beating PBit under sign noise) is consistent with
  their finding that geometry decides which method wins.
- **Karimireddy, Rebjock, Stich, Jaggi, "Error feedback fixes SignSGD and
  other gradient compression schemes" (referenced as the canonical failure
  analysis of sign methods).** Sign methods have known failure modes; our
  Rosenbrock sign row (PBit stably worse) is one of them measured on a toy
  valley.

## Benchmarking methodology

- **Glasserman, Yao, "Some Guidelines and Guarantees for Common Random
  Numbers", Management Science 38(6), 884-908 (1992).** DOI:
  10.1287/mnsc.38.6.884. The CRN classic: shared random streams reduce the
  variance of comparisons between simulated systems. The study's three-stream
  design is an application of this to optimizer benchmarking.
- **Kleinman, Spall, Naiman, "Simulation-Based Optimization with Stochastic
  Approximation Using Common Random Numbers", Management Science 45(11)
  (1999).** CRN inside iterative optimization itself.
- **Sivaprasad, Mai, Vogels, Jaggi, Fleuret, "Optimizer Benchmarking Needs to
  Account for Hyperparameter Tuning", ICML 2020, PMLR 119.**
  arXiv:1910.11758. Argues optimizer comparisons are invalid without an
  accounting of tuning cost, and that learning-rate-only tuning of Adam is the
  most reliable baseline. The tuning protocol here (declared grid, published
  scores, held-out evaluation) follows this argument; our findings that
  hand-tuned shared learning rates distorted every prior claim in this
  repository are a single-codebase instance of their point.

## What is novel here, stated precisely

1. The p-bit/BSN update rule used as a continuous gradient-descent optimizer,
  benchmarked against standard optimizers with annealing, on test landscapes
  and MNIST. The rule is old; this use of it was not found in the literature.
2. A demonstrated benchmark failure mode: per-noise-condition hyperparameter
  tuning turning a "noise effect" into a selection lottery, repaired by
  clean-tune-and-transfer. Not found discussed elsewhere.
3. A demonstrated measurement confound: gradient clipping making a
  quantization experiment unable to measure quantization (clip(quantized) ==
  clip(raw) on every step). Not found discussed elsewhere.
4. An honest negative transfer result: toy-landscape p-bit noise robustness
  does not transfer to MNIST, with a regime-based explanation (wandering
  versus convergence regimes) that predicts both directions.

What is NOT novel: the update rule (Camsari 2017 and the older BSN
literature), CRN itself (Glasserman-Yao 1992), the claim that 1-bit gradients
can work (Bernstein 2018), and the claim that tuning distorts optimizer
comparisons (Sivaprasad 2020). The paper must present items 1-4 as
measurement-and-replication contributions on top of established ground, not
as new physics.
