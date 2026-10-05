"""SIR-Nano model configuration: one object, validated, embeddable in checkpoints.

Everything the model needs comes from config — no hyper-parameter is hard-coded in the modules, so
`configs/*.yaml` alone describes an experiment. The one thing this file adds on top of a plain
dataclass is the **claim guard**: a config that says `target_params: 25M` and silently builds a 6M
model would let a README overstate what was trained, so the actual constructed parameter count is
checked against the target and a >10% drift aborts the run unless `target_params` is null.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class SirModelConfig:
    vocab_size: int = 512
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 352
    max_seq_len: int = 128
    dropout: float = 0.0
    tied_embeddings: bool = True
    norm: str = "pre_layernorm"
    activation: str = "gelu"
    positional: str = "learned_absolute"
    attention: str = "sdpa_causal"
    init_std: float = 0.02
    pad_token_id: int = 0
    eos_token_id: int = 2
    target_params: int | None = None
    dtype: str = "float32"
    extra: dict[str, Any] = field(default_factory=dict)

    # ---- construction -------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SirModelConfig":
        known = {f.name for f in fields(cls)} - {"extra"}
        kwargs = {k: v for k, v in d.items() if k in known and v is not None}
        unknown = {k: v for k, v in d.items() if k not in known}
        # unknown keys are preserved rather than dropped: silently ignoring a typo'd hyper-parameter
        # (e.g. `n_layer`) is how experiments drift from their config
        return cls(**kwargs, extra=dict(unknown))

    @classmethod
    def from_yaml(cls, path: Any) -> "SirModelConfig":
        import yaml
        from pathlib import Path

        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(doc.get("model") or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ---- validation ---------------------------------------------------------------
    def validate(self) -> None:
        if self.vocab_size < 8:
            raise ValueError("vocab_size too small to be a language model vocabulary")
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} must divide by n_heads {self.n_heads}")
        if not 0.0 <= self.dropout < 0.5:
            raise ValueError("dropout must be in [0, 0.5)")
        if self.max_seq_len < 8:
            raise ValueError("max_seq_len must be >= 8")
        if self.norm not in ("pre_layernorm", "pre_rmsnorm", "post_layernorm"):
            raise ValueError(f"unsupported norm {self.norm!r}")
        if self.activation not in ("gelu", "relu", "swiglu"):
            raise ValueError(f"unsupported activation {self.activation!r}")
        if self.positional not in ("learned_absolute", "rope", "none"):
            raise ValueError(f"unsupported positional {self.positional!r}")
        if self.attention not in ("sdpa_causal", "manual_mask"):
            raise ValueError(f"unsupported attention {self.attention!r}")
        for t, name in ((self.pad_token_id, "pad_token_id"), (self.eos_token_id, "eos_token_id")):
            if not 0 <= int(t) < self.vocab_size:
                raise ValueError(f"{name}={t} outside vocabulary {self.vocab_size}")

    # ---- parameter accounting -----------------------------------------------------
    def estimated_params(self) -> dict[str, int]:
        """Analytic parameter count from the config (no magic numbers), for reporting only.

        `training/train.py` always uses the real `num_parameters()` from the built module; this
        estimate exists so a config can be reviewed before a run, and it deliberately reports a
        `vs_actual` gap when someone passes the measured number in.
        """
        v, d, l = self.vocab_size, self.d_model, self.n_layers
        tok_emb = v * d
        pos_emb = self.max_seq_len * d if self.positional == "learned_absolute" else 0
        attn = 4 * d * d  # q, k, v, o projections (no bias counted; biases are negligible and excluded)
        # +biases on the feed-forward (attention projections are bias-free in this model)
        ff = (2 * d * self.d_ff + self.d_ff + d) if self.activation != "swiglu" else (3 * d * self.d_ff)
        norm = (2 * d if self.norm.endswith("layernorm") else d) * (2 * l + 1)
        per_layer = attn + ff + norm
        out_head = 0 if self.tied_embeddings else v * d
        total = tok_emb + pos_emb + l * per_layer + out_head
        return {
            "token_embedding": tok_emb,
            "position_embedding": pos_emb,
            "per_layer": per_layer,
            "layers_total": l * per_layer,
            "output_head": out_head,
            "total": total,
            "grand_total": total,
        }

    def estimate_vs_actual(self, actual_params: int) -> dict[str, Any]:
        e = self.estimated_params()["total"]
        return {"analytic_estimate": int(e), "actual": int(actual_params), "gap": round((e - actual_params) / max(1, actual_params), 5)}

    def check_target(self, actual_params: int) -> dict[str, Any]:
        if self.target_params in (None, 0):
            return {"checked": False, "reason": "target_params is null: no claim to guard"}
        drift = (actual_params - self.target_params) / self.target_params
        report = {
            "checked": True,
            "target_params": int(self.target_params),
            "actual_params": int(actual_params),
            "drift": round(drift, 4),
            "tolerance": 0.10,
            "ok": abs(drift) <= 0.10,
        }
        if not report["ok"]:
            raise ValueError(
                f"model has {actual_params:,} parameters but the config claims a target of "
                f"{self.target_params:,} ({drift:+.1%} drift). Refusing to run: this experiment "
                "would be mislabelled. Either change target_params or change the architecture — "
                "do not change the README."
            )
        return report

    def summary(self) -> str:
        e = self.estimated_params()
        return (
            f"vocab={self.vocab_size} d_model={self.d_model} layers={self.n_layers} heads={self.n_heads} "
            f"ff={self.d_ff} ctx={self.max_seq_len} tied={self.tied_embeddings} "
            f"norm={self.norm} act={self.activation} pos={self.positional} attn={self.attention} "
            f"~params={e['total'] / 1e6:.2f}M"
        )

    def json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
