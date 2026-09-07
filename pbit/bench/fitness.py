"""Fitness helper to wrap a plain function+gradient into a ``FunctionSpec``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pbit.bench.specs import FunctionSpec


def make_fitness(
    name: str,
    dim: int,
    init_bounds: tuple[float, float],
    clip_bounds: tuple[float, float],
    fn: Callable[[Any], float],
    grad: Callable[[Any], Any] | None = None,
    minimum: float | None = None,
    argmin: Any = None,
) -> FunctionSpec:
    """Construct a ``FunctionSpec`` from plain callables."""
    return FunctionSpec(
        name=name,
        dim=dim,
        init_bounds=init_bounds,
        clip_bounds=clip_bounds,
        evaluate=fn,
        grad=grad,
        minimum=minimum,
        argmin=argmin,
    )
