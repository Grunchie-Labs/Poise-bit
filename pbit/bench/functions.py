"""Classic benchmark functions with function-specific domains and analytic gradients.

Each function returns a ``FunctionSpec`` with known minima and *its own*
initialization/clip bounds (no global ``U(-2,2)``):

- ``Rastrigin`` : multi-modal, many local minima; min=0 at x=0.
- ``Ackley``   : deceptive, many local optima; min=0 at x=0. **Analytic gradient**
                 (no slow finite-difference loop).
- ``Rosenbrock``: banana valley; min=0 at x=ones.
- ``Sphere``     : convex quadratic; min=0 at x=0. Gradients are small
                 (``|g| <= 4`` on the default init box), so the runner's
                 gradient clip at 5.0 never fires and quantization
                 experiments on this landscape are unconfounded by clipping.
"""

from __future__ import annotations

import numpy as np

from pbit.bench.specs import FunctionSpec


def _rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    A = 10.0
    return float(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def _rastrigin_grad(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    A = 10.0
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)


def Rastrigin(dim: int = 2, init_bounds: tuple[float, float] = (-5.12, 5.12)) -> FunctionSpec:
    return FunctionSpec(
        name="Rastrigin",
        dim=dim,
        init_bounds=init_bounds,
        clip_bounds=(-5.12, 5.12),
        evaluate=_rastrigin,
        grad=_rastrigin_grad,
        minimum=0.0,
        argmin=np.zeros(dim),
    )


def _ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    a, b, c = 20.0, 0.2, 2 * np.pi
    d = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + np.e)


def _ackley_grad(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    a, b, c = 20.0, 0.2, 2 * np.pi
    d = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    sqrt_s1 = np.sqrt(s1)
    # d/dx_i of -a*exp(-b*sqrt(s1/d))  ;  sqrt(s1/d) = sqrt(s1)/sqrt(d)
    d1 = a * b * np.exp(-b * np.sqrt(s1 / d)) * (x / (max(sqrt_s1 * np.sqrt(d), 1e-12)))
    # d/dx_i of -exp(s2/d)  ->  +exp(s2/d) * c/d * sin(c*x)
    d2 = np.exp(s2 / d) * (c / d) * np.sin(c * x)
    return d1 + d2


def Ackley(dim: int = 2, init_bounds: tuple[float, float] = (-32.768, 32.768)) -> FunctionSpec:
    init_low, init_high = init_bounds
    return FunctionSpec(
        name="Ackley",
        dim=dim,
        init_bounds=(min(init_low, init_high), max(init_low, init_high)),
        clip_bounds=(-32.768, 32.768),
        evaluate=_ackley,
        grad=_ackley_grad,
        minimum=0.0,
        argmin=np.zeros(dim),
    )


def _rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def _rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    for i in range(len(x) - 1):
        g[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
        g[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
    return g


def Rosenbrock(dim: int = 2, init_bounds: tuple[float, float] = (-2.0, 2.0)) -> FunctionSpec:
    return FunctionSpec(
        name="Rosenbrock",
        dim=dim,
        init_bounds=init_bounds,
        clip_bounds=(-5.0, 5.0),
        evaluate=_rosenbrock,
        grad=_rosenbrock_grad,
        minimum=0.0,
        argmin=np.ones(dim),
    )


def _sphere(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(x**2))


def _sphere_grad(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.asarray(x, dtype=np.float64)


def Sphere(dim: int = 2, init_bounds: tuple[float, float] = (-2.0, 2.0)) -> FunctionSpec:
    """Convex quadratic with small gradients.

    On the default init box ``|g_i| = |2 x_i| <= 4.0 < 5.0``, so the
    benchmark runner's gradient clip never fires and quantization noise
    applied before clipping is never erased by it. This is the landscape
    the Rastrigin quantization cell could not be: a cell where a
    quantization comparison actually measures quantization.
    """
    return FunctionSpec(
        name="Sphere",
        dim=dim,
        init_bounds=init_bounds,
        clip_bounds=(-2.0, 2.0),
        evaluate=_sphere,
        grad=_sphere_grad,
        minimum=0.0,
        argmin=np.zeros(dim),
    )


_REGISTRY = {"Rastrigin": Rastrigin, "Ackley": Ackley, "Rosenbrock": Rosenbrock, "Sphere": Sphere}


def get_function(name: str, dim: int = 2) -> FunctionSpec:
    """Lookup a benchmark function by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown function {name!r}; options: {sorted(_REGISTRY)}")
    return _REGISTRY[name](dim)
