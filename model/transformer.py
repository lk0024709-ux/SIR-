"""SIR-Nano: a deliberately plain decoder-only transformer.

v0.1 stays boring on purpose — every component here is textbook so that when a number looks wrong,
the suspicion falls on data and tokenizer rather than on an exotic layer. Specifically:

  * token + learned absolute position embeddings (no RoPE, no ALiBi: fewer ways to be subtly wrong);
  * pre-LayerNorm blocks, multi-head self-attention with an explicit causal mask;
  * GELU feed-forward, residual connections;
  * tied input/output embeddings;
  * next-token cross-entropy with `ignore_index=pad`.

No MoE, no mixture-of-depths, no speculative tricks, no unsupervised contrastive heads. Those are
Phase 2+ decisions to be earned by evidence, not defaults to be imported from a paper.

The causal mask is asserted in tests three ways (shape, no-future-influence, and a padded row),
because a leaky mask produces a *good-looking* loss and a useless model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import SirModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: SirModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.cfg = cfg
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = cfg.dropout
        # cached mask: (1, 1, T, T) boolean, True = allowed
        self.register_buffer("mask", torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).bool(), persistent=False)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        # (b, t, h*dh) -> (b, h, t, dh)
        q = q.view(b, t, self.h, self.dh).transpose(1, 2)
        k = k.view(b, t, self.h, self.dh).transpose(1, 2)
        v = v.view(b, t, self.h, self.dh).transpose(1, 2)
        if self.cfg.attention == "sdpa_causal":
            mask = self.mask[position_offset : position_offset + t, position_offset : position_offset + t]
            mask = mask.view(1, 1, t, t)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.attn_drop if self.training else 0.0, is_causal=False
            )
        else:  # manual_mask: reference path, mathematically identical, used to cross-check SDPA
            scores = (q @ k.transpose(-2, -1)) / (self.dh**0.5)
            allowed = self.mask[:t, :t].view(1, 1, t, t)
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            out = (scores.softmax(dim=-1) @ v)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: SirModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.norm == "pre_rmsnorm":
            self.n1 = RMSNorm(cfg.d_model)
            self.n2 = RMSNorm(cfg.d_model)
        elif cfg.norm == "pre_layernorm":
            self.n1 = nn.LayerNorm(cfg.d_model)
            self.n2 = nn.LayerNorm(cfg.d_model)
        else:  # post_layernorm
            self.n1 = nn.LayerNorm(cfg.d_model)
            self.n2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        if cfg.activation == "swiglu":
            self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
            self.act = None
        else:
            self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
            self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
            self.act = nn.GELU() if cfg.activation == "gelu" else nn.ReLU()
        self.drop = nn.Dropout(cfg.dropout)
        self.post_norm = cfg.norm == "post_layernorm"

    def _ff(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.activation == "swiglu":
            return self.down(F.silu(self.gate(x)) * self.up(x))
        return self.fc2(self.act(self.fc1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.post_norm:
            h = self.n1(x + self.attn(x))
            return self.n2(h + self.drop(self._ff(h)))
        h = self.n1(x)
        x = x + self.drop(self.attn(h))
        h2 = self.n2(x)
        return x + self.drop(self._ff(h2))


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SirNano(nn.Module):
    """Decoder-only transformer with tied embeddings and next-token prediction loss."""

    def __init__(self, cfg: SirModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_emb = (
            nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.positional == "learned_absolute" else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(cfg.d_model) if cfg.norm.endswith("layernorm") else RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tied_embeddings:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        # residual outputs scaled down: standard nanoGPT practice, keeps deep pre-LN stacks stable at init
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight") or name.endswith("down.weight"):
                with torch.no_grad():
                    p.mul_(1.0 / (2 * cfg.n_layers) ** 0.5)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ---- forward ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        logits_to_keep: int | None = None,
    ) -> dict[str, Any]:
        b, t = idx.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds model context {self.cfg.max_seq_len}")
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(t, device=idx.device)).unsqueeze(0)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        keep = t if logits_to_keep is None else min(int(logits_to_keep), t)
        logits = self.head(x[:, t - keep :, :])
        out: dict[str, Any] = {"logits": logits}
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same shape as input ids")
            tgt = targets[:, t - keep :]
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1), ignore_index=self.cfg.pad_token_id
            )
        return out

    def causal_mask_example(self) -> torch.Tensor:
        return self.blocks[0].attn.mask.clone()

    # ---- accounting ---------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> dict[str, int]:
        ps = [p for p in self.parameters() if (p.requires_grad or not trainable_only)]
        total = sum(p.numel() for p in ps)
        emb = self.tok_emb.weight.numel()
        return {
            "total": int(total),
            "embedding": int(emb),
            "non_embedding": int(total - emb),
            "embedding_share": round(emb / max(1, total), 4),
            "tied_embeddings": self.cfg.tied_embeddings,
        }

    @torch.no_grad()
    def estimate_throughput(self, batch_size: int = 8, seq_len: int | None = None, steps: int = 3) -> dict[str, float]:
        """Rough tokens/sec for logging only. Not the benchmark — that is inference/bench_cpu.py."""
        self.eval()
        seq_len = seq_len or self.cfg.max_seq_len
        x = torch.randint(0, self.cfg.vocab_size, (batch_size, seq_len))
        import time

        self(x)  # warm-up
        t0 = time.perf_counter()
        for _ in range(steps):
            self(x)
        dt = (time.perf_counter() - t0) / steps
        return {"forward_tokens_per_sec": round(batch_size * seq_len / dt, 1), "forward_ms": round(dt * 1000, 2)}


def build_from_spec(spec: dict[str, Any]) -> SirNano:
    return SirNano(SirModelConfig.from_dict(spec))
