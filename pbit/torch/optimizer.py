"""Optional PyTorch optimizers.

Importing ``pbit.torch`` requires ``torch`` to be installed (``pip install "pbit[torch]"``).
torch is NOT a runtime dependency of the core package; it is optional and imported
lazily here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

try:  # pragma: no cover - depends on optional torch
    import torch
    from torch.optim import Optimizer
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "pbit.torch requires torch. Install with: pip install 'pbit[torch]'"
    ) from _e


class PBitTorchOptimizer(Optimizer):
    """PyTorch wrapper around the p-bit update rule.

    Applies the probabilistic binary-direction update to each parameter group.
    Supports the same step-size modes as ``PBitOptimizer``. Designed for small /
    low-precision workloads; not claimed to be LLM-scale.

    Parameters
    ----------
    params:
        Iterable of torch parameters or parameter groups (torch.optim API).
    lr, beta0, tau, beta_cap:
        Same cooling hyperparameters as ``PBitOptimizer``.
    step_size:
        ``"proportional"`` (default), ``"constant"``, or ``"floor"``.
    floor:
        Magnitude floor for ``step_size="floor"``.
    seed:
        Seed for the per-parameter Bernoulli sampling.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        beta0: float = 2.0,
        tau: float = 1000.0,
        beta_cap: float = 50.0,
        step_size: str = "proportional",
        floor: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if step_size not in ("proportional", "constant", "floor"):
            raise ValueError(f"unknown step_size {step_size!r}")
        defaults: dict[str, Any] = dict(
            lr=lr, beta0=beta0, tau=tau, beta_cap=beta_cap, step_size=step_size, floor=floor
        )
        super().__init__(params, defaults)
        self._rng = torch.Generator()
        self._t = 0
        if seed is not None:
            self._rng.manual_seed(seed)
        else:
            self._rng.seed()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta0 = group["beta0"]
            tau = group["tau"]
            beta_cap = group["beta_cap"]
            step_size = group["step_size"]
            floor = group["floor"]
            step = self._t + 1
            beta = min(beta0 * (1.0 + step / tau), beta_cap)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                g_scale = g.abs().mean() + 1e-8
                prob = torch.sigmoid(-beta * g / g_scale).to(g.dtype)
                # Bernoulli from a uniform in [-1, 1]; sample in prob space.
                uniform = torch.rand(g.shape, generator=self._rng, device=g.device, dtype=g.dtype)
                sigma = torch.where(uniform < prob, torch.ones_like(g), -torch.ones_like(g))
                abs_g = g.abs()
                if step_size == "constant":
                    mag = torch.full_like(abs_g, lr)
                elif step_size == "floor":
                    mag = lr * (abs_g + floor + 1e-8)
                else:  # proportional
                    mag = lr * (abs_g + 1e-8)
                p.add_(mag * sigma)
            self._t = step
        return loss
