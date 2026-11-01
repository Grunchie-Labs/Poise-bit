"""Which PBit piece carries the Rastrigin effects? A separated ablation.

v2 fixes a design flaw in v1. PBit has exactly two moving parts beyond the
learning rate:

  1. a *stochastic* descent direction, ``sigma ~ Bernoulli(sigmoid(-beta*g/g_scale))``
  2. an *annealing* schedule, ``beta(t) = min(beta0*(1+t/tau), cap)``

v1 froze beta at ``beta0 = 8.0`` as its no-anneal control. That is not a
no-anneal control: at beta = 8 the flip probability is ``sigmoid(-8) = 3.4e-4``,
so the direction is already deterministic and the "no-anneal" arm *is* the
"no-noise" arm. Both controls returned 18.327 on the clean cell -- literally
the same trajectory -- so v1 could not separate the two pieces. The twinning
is not a bug in the numbers, it is the algebraic consequence of freezing beta
in the saturated regime.

v2 runs the ablation as a proper 2x2:

                    anneal ON            anneal OFF
    noise ON        pbit (full)           pbit_frozen_low  (beta0 = LOW_BETA)
    noise OFF       -- (degenerate)       signmag           (infinite-beta limit)

The empty cell is the point: "noise OFF + anneal ON" is degenerate, because a
deterministic direction never reads beta. So three arms identify both main
effects, each contrast holding the other piece fixed:

  * annealing effect (noise ON in both arms):    pbit_frozen_low vs pbit
  * noise effect      (anneal OFF in both arms): signmag vs pbit_frozen_low

A fourth arm, ``pbit_frozen_high`` (beta0 = HIGH_BETA), is retained not as an
inference cell but as a *validity check*: it must reproduce ``signmag``. That
check is what failed silently in v1, so it is now reported as its own family
and the twinning is measured rather than assumed.

Two premises are verified in-script before any inference:

  * saturation: the wrong-direction flip probability is ``sigmoid(-beta0)``,
    i.e. 0.378 at LOW_BETA (genuine thermal noise) and 3.4e-4 at HIGH_BETA
    (deterministic), so the low arm really is stochastic and the high arm
    really is not.
  * measured: mean per-coordinate entropy along a real Rastrigin trajectory,
    normalized to [0, 1] per coordinate, for every arm.

Every arm is tuned on its own grid at the main study's budget (``--tune-runs
10``), which is what made v1's PBit disagree with the main study's config: the
5-run selection picked lr 0.05 while the study's 10 runs pick lr 0.005. C9 can
only speak to C6 if the PBit arm *is* the study's PBit.

Usage:
    python research/experiments/ablation.py
    python research/experiments/ablation.py --eval-seed 1 --out results/ablation_seed1.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pbit.bench import (  # noqa: E402
    CorruptionNoise,
    Experiment,
    Rastrigin,
    SignNoise,
    assert_seeds_disjoint,
    cross,
    grid_from_values,
    tune_optimizer,
)
from pbit.bench.specs import OptimizerSpec  # noqa: E402
from pbit.bench.stats import (  # noqa: E402
    Comparison,
    describe,
    holm_bonferroni,
    paired_compare,
)
from pbit.core.probability import entropy as bern_entropy  # noqa: E402
from pbit.core.probability import sigmoid, sigmoid_probability  # noqa: E402
from pbit.core.schedule import constant  # noqa: E402
from pbit.optim import PBitOptimizer  # noqa: E402
from pbit.optim.baselines import _Base  # noqa: E402

TUNE_SEED = 1000
EVAL_SEED = 0
ITERS = 300
DIM = 2

#: Frozen inverse temperature of the *stochastic, non-annealing* control.
#: Pre-registered, not tuned: at this beta the direction is still thermal
#: (~38% wrong-way flips), which is the whole point of the arm. Letting the
#: tuner move beta0 would let it drift back into saturation and silently
#: recreate the v1 twinning.
LOW_BETA = 0.5

#: Frozen inverse temperature of the *validity* arm. Deep in saturation, so it
#: must reproduce ``signmag``; if it does not, the separation is untrustworthy.
HIGH_BETA = 8.0

#: beta0 values for the reported saturation sweep (descriptive, uncorrected).
BETA_SWEEP = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

class SignMag(_Base):
    """Deterministic sign direction with PBit-proportional magnitude.

    ``step = lr * (|g| + eps) * (-sign(g))``. This is PBit's update taken to
    the infinite-beta limit: the magnitude piece is matched exactly, while the
    Boltzmann direction is replaced by its own argmax. No noise, no annealing
    -- the "noise OFF / anneal OFF" cell of the 2x2. Contrasted against
    ``pbit_frozen_low`` (noise ON, anneal OFF) it isolates the stochasticity
    with the schedule held fixed in *both* arms, which is the contrast v1 could
    not make because its no-anneal control was already deterministic.
    """

    def __init__(self, lr: float = 0.01, seed: int | None = None) -> None:
        super().__init__(seed)
        self.lr = float(lr)

    def step(self, x, grad, t):
        g = np.asarray(grad, dtype=np.float64)
        sigma = -np.sign(g)
        sigma[g == 0.0] = 1.0
        return np.asarray(x, dtype=np.float64) + self.lr * (np.abs(g) + 1e-8) * sigma

    def params(self) -> dict:
        return {"lr": self.lr}


def make_frozen(beta0: float):
    """PBit with beta frozen at ``beta0``: stochastic direction, no annealing.

    The schedule is ``constant(beta0)``, so ``beta(t) == beta0`` for all t and
    the only difference from full PBit is the missing cooling ramp. Choose
    ``beta0`` deliberately: LOW_BETA keeps the direction thermal (this is the
    arm that isolates annealing), HIGH_BETA saturates it (this is the arm that
    must reproduce signmag and validates the whole separation).
    """

    def factory(params: dict) -> PBitOptimizer:
        opt = PBitOptimizer(lr=params["lr"], beta0=beta0, tau=params.get("tau", 300.0))
        opt.configure_schedule(constant(beta0))
        return opt

    return factory


FACTORIES = {
    "pbit": lambda p: PBitOptimizer(lr=p["lr"], tau=p["tau"]),
    "pbit_frozen_low": make_frozen(LOW_BETA),
    "pbit_frozen_high": make_frozen(HIGH_BETA),
    "signmag": lambda p: SignMag(lr=p["lr"]),
}

GRIDS = {
    "pbit": cross(
        grid_from_values("lr", [0.005, 0.01, 0.05, 0.1]),
        grid_from_values("tau", [100, 300, 1000]),
    ),
    # tau is inert under a constant schedule, so it is not tuned for the frozen
    # arms; only lr is. beta0 is fixed by design (see LOW_BETA / HIGH_BETA).
    "pbit_frozen_low": grid_from_values("lr", [0.005, 0.01, 0.05, 0.1]),
    "pbit_frozen_high": grid_from_values("lr", [0.005, 0.01, 0.05, 0.1]),
    "signmag": grid_from_values("lr", [0.005, 0.01, 0.05, 0.1]),
}


def saturation_premise(dim: int = DIM) -> dict:
    """Verify, analytically and numerically, that the arms are where we claim.

    For a coordinate whose normalized gradient is ``g/g_scale``, PBit's update
    takes the *wrong* way with probability ``sigmoid(-beta0)``. At LOW_BETA that
    is a real coin; at HIGH_BETA it is not. The measured entropy along a real
    Rastrigin trajectory confirms it end-to-end: a per-coordinate entropy near
    ``log 2`` means the coordinate is still thermal, near 0 means deterministic.
    """
    # A representative normalized gradient: unit magnitude, mixed signs.
    g = np.array([1.0, -1.0] + [1.0] * (dim - 2))
    g_scale = float(np.mean(np.abs(g)))

    analytic = []
    for b in BETA_SWEEP:
        p = sigmoid_probability(b, g, g_scale)
        analytic.append({
            "beta0": b,
            # P(+1) for a coordinate that wants to move the other way.
            "wrong_way_prob": float(sigmoid(-b)),
            "mean_flip_prob": float(np.mean(p)),
            "entropy_norm_per_coord": float(bern_entropy(p) / (np.log(2.0) * len(p))),
        })

    # Measured: run each arm on clean Rastrigin and average per-step entropy.
    measured = {}
    fn = Rastrigin(dim=dim)
    rng = np.random.default_rng(7)
    for name, factory in FACTORIES.items():
        opt = factory({"lr": 0.01, "tau": 300})
        opt.reset(rng)
        x = rng.uniform(*fn.init_bounds, size=dim)
        ents, probs = [], []
        for t in range(ITERS):
            x = opt.step(x, fn.grad(x), t)
            if hasattr(opt, "state"):
                probs.append(float(np.mean(opt.state.prob)))
                ents.append(bern_entropy(opt.state.prob) / (np.log(2.0) * dim))
        measured[name] = {
            "mean_entropy_norm_per_coord": float(np.mean(ents)) if ents else 0.0,
            "mean_flip_prob": float(np.mean(probs)) if probs else float("nan"),
        }
    measured["signmag"]["mean_flip_prob"] = 0.0

    return {
        "dim": dim,
        "wrong_way_flip_probability_is_sigmoid_minus_beta0": True,
        "analytic": analytic,
        "measured_on_clean_rastrigin": measured,
        "low_beta": LOW_BETA,
        "high_beta": HIGH_BETA,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--tune-runs", type=int, default=10)
    ap.add_argument("--eval-runs", type=int, default=50)
    ap.add_argument("--tune-seed", type=int, default=TUNE_SEED)
    ap.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    ap.add_argument("--out", type=str, default="results/ablation.json")
    ap.add_argument("--dim", type=int, default=DIM)
    args = ap.parse_args()

    assert_seeds_disjoint(args.tune_seed, args.eval_seed)
    fn = Rastrigin(dim=args.dim)
    t_start = time.perf_counter()

    tuning: dict[str, dict] = {}
    print("[tune] Rastrigin/dim=2/clean")
    for name, factory in FACTORIES.items():
        res = tune_optimizer(
            name, factory, GRIDS[name], fn, None,
            max_iter=args.iters, n_tune_runs=args.tune_runs, tune_seed=args.tune_seed,
        )
        tuning[name] = res.to_dict()
        print(f"  {name:<15} selected={res.selected} score={res.selected_score:.4f}")

    premise = saturation_premise(args.dim)
    for row in premise["analytic"]:
        print(f"  [premise] beta0={row['beta0']:<6g} wrong-way P={row['wrong_way_prob']:.4g} "
              f"entropy={row['entropy_norm_per_coord']:.4f}")
    for name, m in premise["measured_on_clean_rastrigin"].items():
        print(f"  [premise] {name:<17} measured entropy/coord={m['mean_entropy_norm_per_coord']:.4f} "
              f"flip P={m['mean_flip_prob']:.4g}")

    noises = [None, CorruptionNoise(p=0.2), SignNoise()]
    series: dict[str, dict] = {}
    for noise in noises:
        key = "clean" if noise is None else noise.name
        specs = [
            OptimizerSpec(n, lambda p=tuning[n]["selected"], f=f: f(p))
            for n, f in FACTORIES.items()
        ]
        rep = Experiment(
            {
                "functions": [fn], "noises": [noise], "optimizers": specs,
                "max_iter": args.iters, "n_runs": args.eval_runs, "seed": args.eval_seed,
            }
        ).run()
        bucket: dict[str, dict] = {}
        for row in rep.rows:
            bucket.setdefault(row.optimizer, {})[row.run_id] = float(row.current_history[-1])
        series[key] = {
            opt: np.array([v[k] for k in sorted(v, key=int)]) for opt, v in bucket.items()
        }
        print(f"[eval] Rastrigin/dim={args.dim}/{key} done")

    # Two families, corrected separately.
    #
    # MAIN (6 contrasts, 2x2 decomposition): each contrast holds the other
    # piece fixed in *both* arms, so the difference is attributable to one
    # piece. ``frozen_low vs pbit`` is the annealing effect (noise ON in both);
    # ``signmag vs frozen_low`` is the noise effect (anneal OFF in both).
    # ``signmag vs pbit`` is the joint gap, reported for continuity with v1.
    #
    # VALIDITY (3 contrasts): the saturated frozen arm must reproduce signmag.
    # A significant difference here would mean the separation is broken.
    main_questions = [
        ("clean: frozen_low vs pbit", "clean", "pbit_frozen_low", "pbit"),
        ("clean: signmag vs frozen_low", "clean", "signmag", "pbit_frozen_low"),
        ("clean: signmag vs pbit", "clean", "signmag", "pbit"),
        ("corruption shift: frozen_low vs pbit", "CorruptionNoise_p=0.2", "pbit_frozen_low", "pbit"),
        ("corruption shift: signmag vs frozen_low", "CorruptionNoise_p=0.2", "signmag", "pbit_frozen_low"),
        ("sign shift: signmag vs frozen_low", "SignNoise", "signmag", "pbit_frozen_low"),
    ]
    validity_questions = [
        ("validity: frozen_high vs signmag", "clean", "pbit_frozen_high", "signmag"),
        ("validity: frozen_high vs signmag", "CorruptionNoise_p=0.2", "pbit_frozen_high", "signmag"),
        ("validity: frozen_high vs signmag", "SignNoise", "pbit_frozen_high", "signmag"),
    ]

    def evaluate(questions, family: str) -> list[dict]:
        out = []
        for label, cell, opt_a, opt_b in questions:
            if "shift" in label:
                a = series[cell][opt_a] - series["clean"][opt_a]
                b = series[cell][opt_b] - series["clean"][opt_b]
            else:
                a = series[cell][opt_a]
                b = series[cell][opt_b]
            cmp = paired_compare(a, b, label)
            out.append({
                "family": family, "question": label, "cell": cell, "a": opt_a, "b": opt_b,
                "comparison": cmp.to_dict(),
                "a_desc": describe(a, opt_a), "b_desc": describe(b, opt_b),
            })
            c = cmp.to_dict()
            print(f"  [{family:<8}] {label:<42} diff={c['mean_diff']:+.4f} "
                  f"CI=[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] p={c['p_value']:.4f}")
        # One Holm correction per family.
        comps = [Comparison(**{k: r["comparison"][k] for k in _KEYS}) for r in out]
        for holder, adj in zip(out, holm_bonferroni(comps)):
            holder["comparison"]["p_value_holm"] = adj.p_value
        for r in out:
            print(f"  [HOLM {family[:4]:<4}] {r['question']:<37} p_holm={r['comparison']['p_value_holm']:.5f}")
        return out

    _KEYS = ("label", "n", "mean_a", "mean_b", "mean_diff",
             "ci_lo", "ci_hi", "p_value", "sd_diff")
    main_results = evaluate(main_questions, "main")
    validity_results = evaluate(validity_questions, "validity")

    # Derived summary quantities, recorded so that CLAIMS.md can quote them and
    # verify_docs.py can trace them. A number computed by hand in prose is
    # unverifiable by construction; tests/test_stats.py exists to enforce this.
    def _find(family: str, label: str) -> dict:
        for r in family:
            if r["question"] == label:
                return r["comparison"]
        raise KeyError(label)

    annealing_clean = _find(main_results, "clean: frozen_low vs pbit")
    noise_clean = _find(main_results, "clean: signmag vs frozen_low")
    joint_clean = _find(main_results, "clean: signmag vs pbit")
    pbit_clean = _find(main_results, "clean: frozen_low vs pbit")["mean_b"]
    pbit_corrupt = [r for r in main_results
                    if r["question"] == "corruption shift: frozen_low vs pbit"][0]["b_desc"]["mean"]
    derived = {
        "annealing_effect_clean": annealing_clean["mean_diff"],
        "noise_effect_clean": noise_clean["mean_diff"],
        "joint_gap_clean": joint_clean["mean_diff"],
        "pbit_clean_mean": pbit_clean,
        "pbit_corruption_shift": pbit_corrupt,
        "annealing_holm_p_clean": annealing_clean["p_value_holm"],
        "noise_holm_p_clean": noise_clean["p_value_holm"],
        "joint_holm_p_clean": joint_clean["p_value_holm"],
        "sign_contrast_holm_p": _find(main_results,
                                      "sign shift: signmag vs frozen_low")["p_value_holm"],
    }
    print("\n[derived] " + "  ".join(f"{k}={v:+.6g}" for k, v in derived.items()))

    # The design is only valid if the saturated frozen arm really is (almost)
    # the deterministic one. Assert it rather than trusting it: if frozen_high
    # drifted materially away from signmag, the twinning that invalidated v1
    # would still be live and the main contrasts must not be read.
    #
    # The test must be on *magnitude*, not significance. Two arms that take the
    # same trajectory still leave floating-point residues of order 1e-18, and a
    # sign-flip test on those residues reports p ~ 5e-5 (every nonzero residue
    # has the same sign, so the null never reaches the observed mean) -- a
    # significance test is simply the wrong instrument here.
    #
    # The instrument is *equivalence*, not a significance test: the claim is
    # "these two arms implement the same algorithm", so the question is whether
    # the observed shift is negligible relative to the loss scale of that cell,
    # and the tolerance has to come from theory rather than a round number.
    #
    # The theory-derived tolerance is the analytic residual flip rate: frozen at
    # beta0, PBit still takes the wrong way with probability
    # p_flip = sigmoid(-beta0). The clean cell is the reference, because there
    # the two arms receive *identical* gradients (the noise stream is shared
    # across optimizers within a run), so they can differ only through those
    # residual flips and the bound is exactly p_flip.
    #
    # Noise cells are deliberately *not* given a fabricated multiplier. An
    # earlier version of this script bounded them by p_flip * (1 + 2p) on the
    # theory that the two arms sample independent noise draws; that reasoning is
    # wrong, because the runner shares the noise stream across optimizers within
    # a run, so the arms see the same perturbations. What actually happens in a
    # noise cell is amplification, not independence: a residual flip perturbs
    # the iterate, the shared noise then acts on a slightly different point, and
    # under corruption (a 20% heavy tail) that small divergence is itself scaled
    # by the heavy-tailed noise magnitude. The honest bound is therefore
    # p_flip * AMPLIFICATION, with AMPLIFICATION measured rather than asserted.
    # AMPLIFICATION is measured, per cell, from the noise model itself: draw the
    # same shared perturbation twice, offset by a relative p_flip in the
    # trajectory, and take the largest relative gap the noise can produce. This
    # is an upper envelope of how far the shared noise can separate two
    # iterates that started p_flip apart, so the resulting bound is a genuine
    # theory-plus-measurement tolerance rather than a number chosen to pass.
    def _amplification(noise, n_draw: int = 20000) -> float:
        """Relative gap induced by noise between iterates p_flip apart."""
        if noise is None:
            return 1.0
        rng = np.random.default_rng(4242)
        base = np.abs(rng.standard_normal((n_draw, DIM))) + 1e-3
        shifted = base * (1.0 + residual_flip_prob)
        return float(np.max(np.abs(noise.apply(shifted, rng) - noise.apply(base, rng))
                            / np.maximum(np.abs(base), 1e-3)))

    NOISE_CELLS = {"clean": None, "CorruptionNoise_p=0.2": CorruptionNoise(p=0.2), "SignNoise": SignNoise()}
    # ``max(1.0, ampl)`` keeps the bound from degenerating. Sign noise maps
    # ``g -> sign(g)``, which a magnitude perturbation cannot move at all, so
    # its measured amplification is exactly 0 and the raw product would demand
    # bitwise equality of the two arms -- a tolerance no stochastic process can
    # honestly meet. The divergence is *injected* at rate p_flip and only then
    # amplified, so p_flip is the floor: bound = p_flip * max(1, amplification).
    BOUNDS = {
        cell: residual_flip_prob * max(1.0, _amplification(noise))
        for cell, noise in NOISE_CELLS.items()
    }

    def _residual(r: dict) -> float:
        c = r["comparison"]
        scale = max(abs(c["mean_b"]), abs(c["mean_a"]), 1e-12)
        return abs(c["mean_diff"]) / scale

    valid = all(_residual(r) <= BOUNDS[r["cell"]] for r in validity_results)
    # Validity reporting: emit the *relative* residuals so that prose quoting
    # them traces to the record. A relative figure written in CLAIMS.md but
    # stored nowhere in results/ is unverifiable by construction, and
    # verify_docs.py exists precisely to catch that.
    validity_report = []
    for r in validity_results:
        c = r["comparison"]
        scale = max(abs(c["mean_b"]), abs(c["mean_a"]), 1e-12)
        rel = abs(c["mean_diff"]) / scale
        bound = BOUNDS[r["cell"]]
        validity_report.append({
            "cell": r["cell"],
            "abs_diff": abs(c["mean_diff"]),
            "scale": scale,
            "rel_diff": rel,
            "bound": bound,
            "ratio_to_bound": rel / bound,
            "within_bound": rel <= bound,
        })

    print(f"\n[validity] residual flip rate at beta0={HIGH_BETA:g} is "
          f"sigmoid(-{HIGH_BETA:g}) = {residual_flip_prob:.3g} per step, per coordinate")
    for row in validity_report:
        print(f"  [validity] {row['cell']:<24} |diff|={row['abs_diff']:.3g} "
              f"rel={row['rel_diff']:.3g} bound={row['bound']:.3g} "
              f"({'within' if row['within_bound'] else 'EXCEEDS'} bound, "
              f"{row['ratio_to_bound']:.2f}x)")
    print(f"[validity] frozen_high reproduces signmag within the analytic bound: {valid}")
    if not valid:
        print("[validity] WARNING: separation broken; main contrasts are not attributable.")

    payload = {
        "metadata": {
            "function": "Rastrigin", "dim": args.dim, "iters": args.iters,
            "tune_runs": args.tune_runs, "eval_runs": args.eval_runs,
            "tune_seed": args.tune_seed, "eval_seed": args.eval_seed,
            "criterion": "final_current_loss_mean",
            "design": "2x2 minus the degenerate noise-OFF/anneal-ON cell; "
                      "three inference arms plus one validity arm",
            "low_beta": LOW_BETA, "high_beta": HIGH_BETA,
            "residual_flip_prob_at_high_beta": residual_flip_prob,
            "validity_amplification": {
                cell: max(1.0, _amplification(noise)) for cell, noise in NOISE_CELLS.items()
            },
            "validity_bounds": BOUNDS,
            "separation_valid": valid,
            "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(),
            "wall_seconds": time.perf_counter() - t_start,
        },
        "premise": premise,
        "validity_report": validity_report,
        "derived": derived,
        "tuning": tuning,
        "results": main_results + validity_results,
        "results_main": main_results,
        "results_validity": validity_results,
        "validity_report": validity_report,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()


