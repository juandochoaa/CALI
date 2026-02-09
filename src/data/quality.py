from __future__ import annotations

from pathlib import Path


def ge_project_path(root: str | Path = '.') -> Path:
    return Path(root) / 'great_expectations'
