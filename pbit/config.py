"""Shared configuration dataclasses for pbit.

All user-facing configuration lives in frozen dataclasses so that configs are
hashable and can be embedded in result metadata / config hashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SeedConfig:
    """Global reproducibility seed.

    Parameters
    ----------
    master_seed:
        Root seed. A stable SHA256-derived RNG per (optimizer, noise, init, run)
        is derived from this value. Never fed directly to numpy's global stream.
    """

    master_seed: int = 42


@dataclass(frozen=True)
class OptimizerConfig:
    """Immutable snapshot of an optimizer's hyperparameters (for reporting).

    This is intentionally a plain key/value container: each optimizer builds
    its own config dict; ``Experiment`` serializes it verbatim into metadata.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}
