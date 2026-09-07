"""Tests for the optional PyTorch optimizer.

These are marked with @pytest.mark.torch and will be deselected when torch is
not installed (or the user runs ``-m 'not torch'``).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from pbit.torch import PBitTorchOptimizer  # noqa: E402


@pytest.mark.torch
def test_tiny_model_loss_decreases():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    opt = PBitTorchOptimizer(model.parameters(), lr=1e-2, tau=1000)
    x = torch.randn(32, 4)
    y = torch.randn(32, 1)
    loss0 = None
    for _ in range(200):
        opt.zero_grad()
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, y)
        if loss0 is None:
            loss0 = loss.item()
        loss.backward()
        opt.step()
    final = torch.nn.functional.mse_loss(model(x), y).item()
    assert final < loss0


@pytest.mark.torch
def test_state_dict_roundtrip():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    opt = PBitTorchOptimizer(model.parameters(), lr=1e-2)
    sd = opt.state_dict()
    opt.load_state_dict(sd)
    assert len(opt.param_groups) == 1


@pytest.mark.torch
def test_seed_reproducible():
    torch.manual_seed(0)
    model1 = torch.nn.Linear(2, 1)
    model2 = torch.nn.Linear(2, 1)
    model2.load_state_dict(model1.state_dict())  # identical init

    opt1 = PBitTorchOptimizer(model1.parameters(), lr=0.1, seed=7)
    opt2 = PBitTorchOptimizer(model2.parameters(), lr=0.1, seed=7)
    x = torch.tensor([[1.0, -1.0]])

    model1.zero_grad()
    (model1(x) * 2.0).sum().backward()
    opt1.step()

    model2.zero_grad()
    (model2(x) * 2.0).sum().backward()
    opt2.step()

    p1 = [p.detach().numpy() for p in model1.parameters()]
    p2 = [p.detach().numpy() for p in model2.parameters()]
    for a, b in zip(p1, p2):
        assert np.allclose(a, b)
