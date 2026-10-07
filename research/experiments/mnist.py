"""MNIST study: does the toy-landscape story transfer to a real task?

Trains a small MLP on MNIST with PBit, Adam, and SGD, under clean gradients
and under 1-bit sign-only gradients (the C2 condition on a real task). The
protocol mirrors the toy study: tune each optimizer's learning rate on a
validation split under clean gradients, transfer the selected configuration to
the sign condition, evaluate on held-out test data over multiple runs, and
compare in pairs.

Pairing: run ``s`` gives every optimizer the same weight init and the same data
order, so within-run differences cancel both. PBit's Bernoulli sampling is
seeded per run.

Usage:
    python research/experiments/mnist.py
    python research/experiments/mnist.py --runs 15 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torchvision  # noqa: E402
from torch.nn.functional import cross_entropy  # noqa: E402

from pbit.bench.stats import describe, paired_compare  # noqa: E402
from pbit.torch import PBitTorchOptimizer  # noqa: E402

GRIDS = {
    "pbit": [(lr, tau) for lr in (1e-3, 1e-2, 5e-2) for tau in (100.0, 1000.0)],
    "adam": [(lr,) for lr in (1e-3, 1e-2)],
    "sgd": [(lr,) for lr in (1e-2, 1e-1, 5e-1)],
}


def make_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    )


def make_optimizer(name: str, params, config, seed: int):
    if name == "pbit":
        lr, tau = config
        return PBitTorchOptimizer(params, lr=lr, tau=tau, seed=seed)
    if name == "adam":
        (lr,) = config
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        (lr,) = config
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(name)


def batch_iter(x, y, batch: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    for start in range(0, x.shape[0], batch):
        idx = perm[start:start + batch]
        yield x[idx], y[idx]


def train_once(name: str, config, seed: int, epochs: int, sign_only: bool,
               x_tr, y_tr, x_te, y_te, batch: int = 256) -> dict:
    model = make_model(seed)
    opt = make_optimizer(name, model.parameters(), config, seed)
    for epoch in range(epochs):
        model.train()
        for xb, yb in batch_iter(x_tr, y_tr, batch, seed + epoch):
            opt.zero_grad()
            loss = cross_entropy(model(xb), yb)
            loss.backward()
            if sign_only:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.sign_()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(x_te)
        test_loss = float(cross_entropy(logits, y_te))
        test_acc = float((logits.argmax(1) == y_te).float().mean())
    return {"test_loss": test_loss, "test_acc": test_acc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--tune-runs", type=int, default=3)
    ap.add_argument("--tune-epochs", type=int, default=5)
    ap.add_argument("--out", type=str, default="results/mnist.json")
    args = ap.parse_args()

    train_ds = torchvision.datasets.MNIST(
        root="out/_mnist", train=True, download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_ds = torchvision.datasets.MNIST(
        root="out/_mnist", train=False, download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    # Train on the first 50k; the last 10k of train is the tuning split.
    x_tr = train_ds.data[:50000].float() / 255.0
    y_tr = train_ds.targets[:50000]
    x_val = train_ds.data[50000:].float() / 255.0
    y_val = train_ds.targets[50000:]
    x_te = test_ds.data.float() / 255.0
    y_te = test_ds.targets

    # Tune on the validation split, clean gradients only, then transfer.
    selected: dict[str, tuple] = {}
    for name, grid in GRIDS.items():
        best, best_loss = None, float("inf")
        for config in grid:
            losses = [
                train_once(name, config, seed=100 + r, epochs=args.tune_epochs,
                           sign_only=False, x_tr=x_tr, y_tr=y_tr,
                           x_te=x_val, y_te=y_val)["test_loss"]
                for r in range(args.tune_runs)
            ]
            mean_loss = float(np.mean(losses))
            print(f"[tune] {name:<6} {config} -> val loss {mean_loss:.4f}")
            if mean_loss < best_loss:
                best, best_loss = config, mean_loss
        selected[name] = best
        print(f"[tune] {name:<6} selected {best}")

    # Evaluate on test data, clean and sign-only, paired by run seed.
    results: dict[str, dict[str, list]] = {
        name: {"clean": [], "sign": []} for name in GRIDS
    }
    for r in range(args.runs):
        seed = 1000 + r
        for name in GRIDS:
            for cond, sign_only in (("clean", False), ("sign", True)):
                out = train_once(name, selected[name], seed=seed, epochs=args.epochs,
                                 sign_only=sign_only, x_tr=x_tr, y_tr=y_tr,
                                 x_te=x_te, y_te=y_te)
                results[name][cond].append(out)
        print(f"[eval] run {r + 1}/{args.runs} done")

    def metric(name, cond, key):
        return np.array([o[key] for o in results[name][cond]])

    comparisons = []
    for key in ("test_loss", "test_acc"):
        for other in ("adam", "sgd"):
            cmp = paired_compare(metric("pbit", "clean", key),
                                 metric(other, "clean", key),
                                 f"PBit vs {other} ({key})")
            comparisons.append({"metric": key, "comparison": cmp.to_dict(),
                                "question": f"clean {key}: PBit vs {other}"})
    # Sign-noise degradation: shift (sign - clean) per optimizer, compared.
    for key in ("test_loss", "test_acc"):
        pbit_shift = metric("pbit", "sign", key) - metric("pbit", "clean", key)
        for other in ("adam", "sgd"):
            other_shift = metric(other, "sign", key) - metric(other, "clean", key)
            cmp = paired_compare(pbit_shift, other_shift,
                                 f"sign shift: PBit vs {other} ({key})")
            comparisons.append({
                "metric": key,
                "comparison": cmp.to_dict(),
                "question": f"sign-noise shift on {key}: PBit vs {other}",
                "pbit_shift": describe(pbit_shift, "PBit shift"),
                "other_shift": describe(other_shift, f"{other} shift"),
            })

    payload = {
        "metadata": {
            "runs": args.runs,
            "epochs": args.epochs,
            "tune_runs": args.tune_runs,
            "tune_epochs": args.tune_epochs,
            "train_samples": int(x_tr.shape[0]),
            "val_samples": int(x_val.shape[0]),
            "test_samples": int(x_te.shape[0]),
            "model": "784-128-10 MLP, ReLU",
            "torch": torch.__version__,
        },
        "selected": {k: list(v) for k, v in selected.items()},
        "per_run": results,
        "summaries": {
            name: {
                cond: {key: describe(metric(name, cond, key), f"{name} {cond} {key}")
                       for key in ("test_loss", "test_acc")}
                for cond in ("clean", "sign")
            }
            for name in GRIDS
        },
        "comparisons": comparisons,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== summaries (mean test metrics) ===")
    for name in GRIDS:
        for cond in ("clean", "sign"):
            for key in ("test_loss", "test_acc"):
                d = payload["summaries"][name][cond][key]
                print(f"  {name:<6} {cond:<6} {key:<9} mean={d['mean']:.4f} sd={d['sd']:.4f}")
    print("\n=== comparisons ===")
    for c in comparisons:
        cmp = c["comparison"]
        print(f"  {c['question']:<42} diff={cmp['mean_diff']:+.4f} "
              f"CI=[{cmp['ci_lo']:+.4f},{cmp['ci_hi']:+.4f}] p={cmp['p_value']:.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
