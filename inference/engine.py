"""Minimal CPU inference for SIR-Nano.

Deliberately unoptimised in exactly one way, and honest about it: there is no KV cache. Each new
token re-runs the whole visible sequence, so cost is O(t^2) in generated length. At v0.1 sizes
(ctx 64-512, ~1M params) that is a fraction of a second and it keeps the sampling path tiny enough to
audit; the cache is a Phase 7 edge requirement and is tracked as such in docs/architecture.md rather
than being quietly deferred.

Sampling is greedy by default (temperature 0). Greedy is the default because the acceptance tests need
*reproducible* text: `do_sample: false` twice must give the same string.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GenSettings:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    eos_ids: tuple[int, ...] = ()
    pad_id: int = 0
    seed: int | None = None
    ban_repetition_beyond: int = 0  # 0 = off; when set, blocks a token that would extend an n-gram loop

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0


def load_for_inference(checkpoint: Path | str, device: str = "cpu", allow_tokenizer_change: bool = False):
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from training import checkpoint as ckpt_mod

    model, state = ckpt_mod.load(Path(checkpoint), map_location=device)
    tok = ckpt_mod.load_tokenizer_from_checkpoint(state) if not allow_tokenizer_change else _load_lenient(state)
    if model is None:
        raise RuntimeError("checkpoint did not contain a buildable model config")
    model.to(device).eval()
    return model, tok, state


def _load_lenient(state: dict[str, Any]):
    import json as _json
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from tokenizer.api import SirTokenizer

    spec = (_state_tokenizer_spec(state) or "")
    if not spec:
        raise RuntimeError("no tokenizer spec recorded in checkpoint")
    return SirTokenizer.load(Path(spec))


def _state_tokenizer_spec(state: dict[str, Any]) -> str | None:
    return (state.get("tokenizer") or {}).get("spec")


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    settings: GenSettings | None = None,
    stream_print: bool = False,
) -> dict[str, Any]:
    s = settings or GenSettings()
    if s.seed is not None:
        g = torch.Generator().manual_seed(int(s.seed))
    else:
        g = None
    device = next(model.parameters()).device
    ctx = int(model.cfg.max_seq_len)
    ids = list(tokenizer.encode(prompt))
    if s.eos_ids:
        ids = ids + [min(s.eos_ids)]  # condition the model on the end-of-document token it was trained with
    if not ids:
        return {"text": "", "prompt_tokens": 0, "generated_tokens": 0, "truncated": False, "degenerate": True, "note": "empty encoding"}
    prompt_len = len(ids)
    if prompt_len > ctx:
        ids = ids[-ctx:]  # right-context is kept: the tail is what the model must complete
    eos = set(int(i) for i in (s.eos_ids or (model.cfg.eos_token_id,)))
    t0 = time.perf_counter()
    out: list[int] = []
    stopped = None
    for _ in range(int(s.max_new_tokens)):
        window = torch.tensor([ids[-ctx:]], dtype=torch.long, device=device)
        logits = model(window, logits_to_keep=1)["logits"][0, -1].float()
        nxt = _pick(logits, s, g)
        if s.ban_repetition_beyond and _would_loop(ids + [nxt], s.ban_repetition_beyond):
            logits[nxt] = -float("inf")
            nxt = _pick(logits, s, g)
        if nxt in eos:
            stopped = "eos"
            break
        ids.append(nxt)
        out.append(nxt)
        if stream_print:
            print(tokenizer.decode([nxt]), end="", flush=True)
    dt = time.perf_counter() - t0
    text = tokenizer.decode(out)
    return {
        "text": text,
        "prompt": prompt,
        "prompt_tokens": prompt_len,
        "generated_tokens": len(out),
        "seconds": round(dt, 4),
        "tokens_per_sec": round(len(out) / max(1e-9, dt), 2),
        "ms_per_token": round(1000 * dt / max(1, len(out)), 3),
        "first_token_ms": None,
        "truncated": prompt_len > ctx,
        "stopped": stopped,
        "eos_id": model.cfg.eos_token_id,
        "context_length": ctx,
        "degenerate": is_degenerate(text, out),
    }


def _pick(logits: torch.Tensor, s: GenSettings, g: torch.Generator | None) -> int:
    if not s.do_sample:
        return int(torch.argmax(logits).item())
    tmp = max(1e-6, float(s.temperature))
    probs = torch.softmax(logits / tmp, dim=-1)
    if s.top_k and s.top_k > 0:
        k = min(int(s.top_k), probs.numel())
        v, i = torch.topk(probs, k)
        mask = torch.zeros_like(probs)
        mask.scatter_(0, i, v)
        probs = mask
        probs = probs / probs.sum()
    if s.top_p < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = sorted_p.cumsum(0)
        keep = cum <= s.top_p
        keep[0] = True
        mask = torch.zeros_like(probs)
        mask[sorted_i[keep]] = sorted_p[keep]
        probs = mask / mask.sum().clamp_min(1e-12)
    return int(torch.multinomial(probs, 1, generator=g).item())


def _would_loop(ids: list[int], max_run: int) -> bool:
    """True when the last `max_run` tokens are a single repeated token (degenerate loop)."""
    tail = ids[-max_run:]
    return len(tail) == max_run and len(set(tail)) == 1


def tail_cycle_cover(ids: list[int], max_period: int = 16) -> tuple[int, int, int]:
    """Longest run of a repeating token cycle at the end of a generation.

    Returns (period, repetitions, tokens_covered). A model stuck in a loop shows a small period
    repeated many times; a model producing real text does not. This catches the case that an
    n-gram-frequency test misses: short cycles like "नियमा नियमा नियमा".
    """
    best = (0, 0, 0)
    n = len(ids)
    for period in range(1, max_period + 1):
        if n < period * 3:
            continue
        cycle = ids[-period:]
        reps = 0
        i = n
        while i >= period and ids[i - period : i] == cycle:
            reps += 1
            i -= period
        if reps >= 3 and reps * period > best[2]:
            best = (period, reps, reps * period)
    return best


def is_degenerate(
    text: str,
    ids: list[int],
    ngram: int = 8,
    max_run_tokens: int = 32,
    min_unique_ngram_ratio: float = 0.35,
    max_tail_cycle_cover: float = 0.5,
    max_char_dominance: float = 0.6,
) -> bool:
    """Mechanical degeneracy test shared by inference, the probe harness and the tests.

    A generation counts as degenerate when ANY of:
      * it is empty or whitespace only;
      * one token id repeats for `max_run_tokens` straight tokens;
      * an n-gram of `ngram` tokens repeats back-to-back at least three times;
      * a short token cycle (period <= 16) covers more than `max_tail_cycle_cover` of the output;
      * fewer than `min_unique_ngram_ratio` of its 4-grams are distinct (phrase-level looping);
      * a single non-space character makes up more than `max_char_dominance` of the text
        ("------", "!!!!", a model discovering one easy glyph and hammering it).
    """
    if not text or not text.strip():
        return True
    if len(ids) >= max_run_tokens and len(set(ids[-max_run_tokens:])) == 1:
        return True
    if len(ids) >= ngram * 3:
        last = tuple(ids[-ngram:])
        repeats = 0
        for i in range(len(ids) - ngram, -1, -ngram):
            if tuple(ids[i : i + ngram]) == last:
                repeats += 1
            else:
                break
        if repeats >= 3:
            return True
    if len(ids) >= 6:
        _period, _reps, covered = tail_cycle_cover(ids)
        if covered / len(ids) > max_tail_cycle_cover:
            return True
        grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
        if len(grams) >= 6 and len(set(grams)) / len(grams) < min_unique_ngram_ratio:
            return True
    body = text.replace(" ", "").replace("\n", "")
    if len(body) >= 12:
        top = max(body.count(c) for c in set(body))
        if top / len(body) > max_char_dominance:
            return True
    return False


def perplexity_from_ids(model, ids: list[int], pad_id: int, device: str = "cpu", seq_len: int | None = None) -> dict[str, float]:
    """Teacher-forced mean NLL / perplexity over an explicit token id list (used by probes/tests)."""
    ctx = int(seq_len or model.cfg.max_seq_len)
    if len(ids) < 3:
        return {"nll": float("nan"), "perplexity": float("nan"), "tokens": len(ids)}
    total, count = 0.0, 0
    for start in range(0, len(ids) - 1, ctx):
        chunk = ids[start : start + ctx + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        with torch.no_grad():
            loss = model(x, targets=y)["loss"]
        n = int((y != pad_id).sum().item())
        if n == 0:
            continue
        total += float(loss.item()) * n
        count += n
    nll = total / max(1, count)
    return {"nll": round(nll, 6), "perplexity": round(math.exp(min(80.0, nll)), 4), "tokens": count}
