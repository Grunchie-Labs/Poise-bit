"""pbit: probabilistic p-bit optimization and honest benchmarking.

Public API is re-exported here for convenience:

- ``pbit.optim`` : PBitOptimizer and baselines
- ``pbit.bench`` : Experiment / Report / compare_reports
- ``pbit.core``  : probability, schedule, rng primitives
- ``pbit.torch`` : optional PyTorch wrapper (requires ``pbit[torch]``)
"""

from pbit._version import __version__

__all__ = ["__version__"]
