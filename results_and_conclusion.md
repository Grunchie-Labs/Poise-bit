# Results moved

The benchmark tables that used to live here were withdrawn.

They reported bare means with no uncertainty, and two of the conditions behind
them were not equal between optimizers. The ask/tell optimizers received twice
the objective-evaluation budget of every other method, and EvolutionStrategy was
compared at equal iteration count despite needing ten evaluations per update.

The claims those tables made have been remeasured under a protocol that reports
confidence intervals, compares optimizers in pairs on shared random streams, and
assigns verdicts from the data. Three of the five registered claims did not
resolve, and one resolved against p-bit optimization.

The current results are in [`research/CLAIMS.md`](research/CLAIMS.md), with the
method in [`research/METHOD.md`](research/METHOD.md) and the machine-readable
records in `results/`. Regenerate them with:

```bash
python research/experiments/claims.py --seeds 30 --iters 400
python research/experiments/power.py --iters 300 --draws 2000
```