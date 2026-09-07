"""pytorch_mlp.py — train a tiny MLP with the optional PBit torch optimizer.

Requires torch: ``pip install "pbit[torch]"``.

Trains a small binary classifier on synthetic data using PBitTorchOptimizer.
This is a demo of the integration, not a claim about LLM-scale training.

Run:
    python examples/pytorch_mlp.py
"""

from __future__ import annotations

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from pbit.torch import PBitTorchOptimizer


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_data(n: int = 600, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    y = ((x[:, 0] * x[:, 1] > 0).float()).unsqueeze(1)
    return x, y


def main() -> None:
    torch.manual_seed(0)
    x, y = make_data()

    # Train one model with PBit and one with SGD for comparison.
    models = {"pbit": TinyMLP(), "sgd": TinyMLP()}
    opt_pbit = PBitTorchOptimizer(models["pbit"].parameters(), lr=0.05, beta0=2.0, tau=300, seed=1)
    opt_sgd = torch.optim.SGD(models["sgd"].parameters(), lr=0.05)

    steps = 400
    for step in range(1, steps + 1):
        for opt, model in ((opt_pbit, models["pbit"]), (opt_sgd, models["sgd"])):
            opt.zero_grad()
            logits = model(x)
            loss = binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
        if step in (1, 50, 100, 200, 400):
            p, s = models["pbit"], models["sgd"]
            lp = binary_cross_entropy_with_logits(p(x), y).item()
            ls = binary_cross_entropy_with_logits(s(x), y).item()
            print(f"step {step:>4}  pbit={lp:.4f}  sgd={ls:.4f}")

    print("\nDone. Final auroc-style accuracy:")
    for name, m in models.items():
        with torch.no_grad():
            pred = (torch.sigmoid(m(x)) > 0.5).float()
            acc = (pred == y).float().mean().item()
        print(f"  {name:<5} accuracy = {acc:.3f}")


if __name__ == "__main__":
    main()
