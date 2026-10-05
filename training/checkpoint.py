"""Checkpoint format: a file must be enough to reproduce inference without reading a README.

Every SIR checkpoint carries, next to the weights:
  model config · tokenizer spec path + artifact sha256 + kind + vocab size · global step · seed
  · best/last validation loss · git commit · library versions · the *resolved* run config.

Checkpoints are large and therefore git-ignored (`*.pt`, `runs/` in .gitignore). SIR does not use
Git LFS, so nothing here should ever be committed; `assert_not_tracked()` makes that a test failure
rather than a bad day for the repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1


def environment_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_threads": torch.get_num_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "short": run("rev-parse", "--short", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def sha256_file(path: Path) -> str:
    if not Path(path).exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_info(spec_path: Path | None) -> dict[str, Any]:
    if not spec_path or not Path(spec_path).exists():
        return {"spec": None, "note": "no tokenizer spec supplied — inference from this checkpoint is not reproducible"}
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    artifact = Path(spec_path).parent / Path(spec["path"]).name
    return {
        "spec": str(spec_path),
        "spec_sha256": sha256_file(Path(spec_path)),
        "kind": spec.get("kind"),
        "vocab_size_declared": spec.get("vocab_size"),
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "special_ids": {k: spec.get(k) for k in ("pad_id", "unk_id", "bos_id", "eos_id")},
    }


def save(
    path: Path | str,
    model: torch.nn.Module,
    *,
    model_cfg: dict[str, Any],
    step: int,
    seed: int | None,
    val_loss: float | None,
    run_config: dict[str, Any] | None = None,
    tokenizer_spec: Path | str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "sir_checkpoint_format": FORMAT_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": model_cfg,
        "step": int(step),
        "seed": seed,
        "validation_loss": None if val_loss is None else float(val_loss),
        "tokenizer": tokenizer_info(Path(tokenizer_spec) if tokenizer_spec else None),
        "git": git_info(),
        "environment": environment_info(),
        "run_config": run_config or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state["scheduler_state"] = scheduler.state_dict()
    torch.save(state, path)
    size = path.stat().st_size
    return {"path": str(path), "bytes": size, "sha256": sha256_file(path), "step": state["step"]}


def load(path: Path | str, *, map_location: str = "cpu", strict: bool = True, build_model: bool = True):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(state, dict) or "model_state" not in state:
        raise ValueError(f"{path} is not a SIR checkpoint (missing model_state)")
    fmt = state.get("sir_checkpoint_format")
    if fmt != FORMAT_VERSION:
        print(f"warning: checkpoint format {fmt!r} != expected {FORMAT_VERSION}", file=sys.stderr)
    model = None
    if build_model:
        sys.path.insert(0, str(REPO_ROOT))
        from model.config import SirModelConfig
        from model.transformer import SirNano

        cfg = SirModelConfig.from_dict(state["model_config"])
        model = SirNano(cfg)
        model.load_state_dict(state["model_state"], strict=strict)
        model.eval()
    return model, state


def load_tokenizer_from_checkpoint(state: dict[str, Any]):
    """Rebuild the exact tokenizer the checkpoint was trained with (refuses to guess)."""
    sys.path.insert(0, str(REPO_ROOT))
    from tokenizer.api import SirTokenizer

    info = state.get("tokenizer") or {}
    spec_path = info.get("spec")
    if not spec_path:
        raise RuntimeError(
            "checkpoint does not record a tokenizer spec. Refusing to guess one: ids decoded with a "
            "different vocabulary would produce text that looks plausible and means nothing."
        )
    tok = SirTokenizer.load(Path(spec_path))
    recorded = info.get("artifact_sha256")
    if recorded and recorded not in ("missing",) and tok and getattr(tok, "spec", None) is not None:
        art = Path(spec_path).parent / Path(json.loads(Path(spec_path).read_text(encoding='utf-8'))["path"]).name
        now = sha256_file(art)
        if now != recorded:
            raise RuntimeError(
                f"tokenizer artifact changed after training: recorded {recorded[:12]}…, now {now[:12]}… "
                f"({art}). This checkpoint cannot be reproduced with the current tokenizer files."
            )
    return tok


def rotate(run_dir: Path | str, keep_last: int = 3, prefix: str = "step") -> list[str]:
    """Keep the N newest step checkpoints; `best.pt` is never rotated away."""
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob(f"{prefix}-*.pt"), key=lambda p: p.stat().st_mtime)
    removed = []
    while len(files) > max(1, int(keep_last)):
        f = files.pop(0)
        f.unlink(missing_ok=True)
        removed.append(str(f))
    return removed


def assert_not_tracked(path: Path | str) -> None:
    """Guard used by tests: a checkpoint in `git status` means .gitignore stopped working."""
    p = Path(path).resolve()
    try:
        rel = p.relative_to(REPO_ROOT)
    except ValueError:
        return
    res = subprocess.run(["git", "check-ignore", "-q", "--", str(rel)], cwd=REPO_ROOT, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"{rel} is a checkpoint path that git would track — .gitignore must cover it")
