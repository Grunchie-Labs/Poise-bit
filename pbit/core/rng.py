"""Stable, non-global random-number generation.

pbit never touches ``numpy.random.seed`` or the global stream. Instead every
consumers' randomness comes from an explicitly passed ``np.random.Generator``
derived deterministically from a ``master_seed`` plus string tags using a
SHA256 hash (never Python's unstable ``hash()``).

Three streams are used by the benchmark runner so that a change in one
optimizer's private randomness cannot perturb the shared start-position or
noise streams:

- ``rng_init``  : shared across optimizers -> paired (common random numbers) starts
- ``rng_noise`` : shared across optimizers -> comparable noisy gradients
- ``rng_opt``   : private per-optimizer randomness
"""

from __future__ import annotations

import hashlib

import numpy as np

_HASH_SEP = "\x1f"


def derive_seed(master_seed: int, *tags: str) -> int:
    """Deterministic integer seed from ``master_seed`` and string ``tags``.

    Uses SHA256 over the joined, tagged payload (no Python ``hash()``), then mixes
    the digest into ``master_seed`` with XOR. Stable across processes/platforms.
    """
    payload = _HASH_SEP.join(str(t) for t in tags)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") ^ int(master_seed) & 0xFFFFFFFFFFFFFFFF


def derive_rng(master_seed: int, *tags: str) -> np.random.Generator:
    """Return an independent ``np.random.Generator`` derived from seeds/tags."""
    seed = derive_seed(master_seed, *tags)
    return np.random.default_rng(np.random.SeedSequence(seed))


def constructor_rng(seed: int | None) -> np.random.Generator:
    """RNG for standalone optimizer use.

    If ``seed is None``, draws entropy from the OS using ``np.random.default_rng()``
    (non-reproducible). Otherwise returns a reproducible ``Generator`` seeded
    directly (a single seed is context-free enough to not need tags).
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)
