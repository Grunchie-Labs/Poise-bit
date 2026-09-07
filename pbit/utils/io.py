"""IO helpers: atomic, ASCII-safe writes for benchmark artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """Write text atomically (write temp file then rename) to avoid partial artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def write_json_atomic(path: str | Path, data: Any) -> Path:
    return atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True))
