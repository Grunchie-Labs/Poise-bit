"""Core primitives for pbit: probability, schedules, and deterministic RNG."""

from pbit.core.probability import bernoulli_bit, entropy, sigmoid, sigmoid_probability
from pbit.core.rng import constructor_rng, derive_rng, derive_seed
from pbit.core.schedule import Schedule, constant, linear_cooling

__all__ = [
    "bernoulli_bit",
    "constructor_rng",
    "constant",
    "derive_rng",
    "derive_seed",
    "entropy",
    "linear_cooling",
    "Schedule",
    "sigmoid",
    "sigmoid_probability",
]
