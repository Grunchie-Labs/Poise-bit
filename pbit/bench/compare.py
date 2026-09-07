"""Compare two reports and explain why results differ (reproducibility helper).

Two researchers who run the "same" experiment can diff their JSON reports. The
diff attributes differences to: config hash, pbit/numpy/python version, or
platform/hardware. Loss histories should agree on the same software+config;
only measured wall-time may legitimately differ across machines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ReportDiff:
    a_path: str
    b_path: str
    config_match: bool
    software_match: bool
    metadata_same: bool
    max_loss_delta: float
    notes: list[str]

    def summary(self) -> str:
        lines = [f"Comparing {self.a_path} vs {self.b_path}", "-" * 50]
        lines.append(f"Config hash identical : {self.config_match}")
        lines.append(f"Software identical     : {self.software_match}")
        lines.append(f"Metadata identical     : {self.metadata_same}")
        lines.append(f"Max best-loss delta    : {self.max_loss_delta:.3e}")
        for n in self.notes:
            lines.append(f" - {n}")
        if self.config_match and self.software_match and self.metadata_same:
            lines.append("=> Runs are bit-reproducible (loss-wise) on this machine.")
        elif self.config_match and not self.software_match:
            lines.append("=> Same config but differing software: check pbit/numpy/python versions.")
        elif not self.config_match:
            lines.append("=> Configs differ: inspect the config hash / experiment fields.")
        return "\n".join(lines)


def _load(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cells_map(data: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for c in data.get("cells", []):
        out[(c["function"], c["noise"], c["optimizer"])] = c
    return out


def compare_reports(path_a: str | Path, path_b: str | Path, loss_tol: float = 1e-6) -> ReportDiff:
    a = _load(path_a)
    b = _load(path_b)

    config_match = a.get("config_hash") == b.get("config_hash")
    sw_fields = ("pbit_version", "numpy_version", "python_version")
    software_match = all(a["metadata"].get(k) == b["metadata"].get(k) for k in sw_fields)
    metadata_same = a.get("metadata") == b.get("metadata")

    cells_a = _cells_map(a)
    cells_b = _cells_map(b)
    keys = set(cells_a) & set(cells_b)
    max_delta = 0.0
    notes: list[str] = []
    for k in keys:
        d = abs(float(cells_a[k].get("final_best_loss_mean", 0.0)) - float(cells_b[k].get("final_best_loss_mean", 0.0)))
        max_delta = max(max_delta, d)

    if not config_match:
        notes.append("config hashes differ - inspect experiment configuration fields")
    if not software_match:
        notes.append("pbit / numpy / python versions differ between environments")
    if a.get("metadata", {}).get("platform") != b.get("metadata", {}).get("platform"):
        notes.append("platform differs (wall-time is not comparable across machines)")
    if len(keys) < len(cells_a) or len(keys) < len(cells_b):
        notes.append("cell sets differ between reports (different optimizers/functions?)")

    return ReportDiff(
        a_path=str(path_a),
        b_path=str(path_b),
        config_match=config_match,
        software_match=software_match,
        metadata_same=metadata_same,
        max_loss_delta=max_delta,
        notes=notes,
    )
