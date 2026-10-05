"""Tiny shared path/config helpers for SIR's command-line tools.

Kept as a root-level module (not a package) on purpose: every tool needs repo-relative paths and
nothing else, and importing a heavy common package from `tokenizer/` into `training/` would create
a dependency that the module boundaries in README.md do not allow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent


def resolve(path: Path | str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def rel(path: Path | str) -> str:
    """Repo-relative display path, or the absolute path when it lives outside the repo.

    Never raises: a display helper that crashes a training run would be a bad trade.
    """
    p = Path(path).resolve() if not Path(path).is_absolute() else Path(path)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def load_config(path: Path | str) -> dict[str, Any]:
    """Read a YAML (or JSON) config into a dict. Missing file is an error; an empty file is {}.

    Configs are the single source of truth for experiments; training/checkpoint.py embeds the
    resolved config into each checkpoint so an artifact can always be traced back to it.
    """
    import yaml

    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    if p.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    raise ValueError(f"unsupported config format: {p.suffix} (use .yaml or .json)")


def cfg_get(cfg: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    sec = cfg.get(section) or {}
    if not isinstance(sec, dict):
        return default
    val = sec.get(key, default)
    return default if val is None and default is not None else val


def write_json(path: Path | str, payload: dict[str, Any]) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return p
