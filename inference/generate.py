"""Text completion from a SIR-Nano checkpoint.

    python -m inference.generate --checkpoint runs/sir_nano_smoke/final.pt --prompt "भारत"

The output banner is part of the tool, not decoration: a ~1M-parameter model trained on a small
fixture corpus produces *plausible-looking* text, and plausible-looking text from a prototype is the
easiest way for this project to overstate itself. Every invocation states what it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inference.engine import GenSettings, generate, load_for_inference  # noqa: E402

BANNER = (
    "SIR-Nano prototype output — a research artifact, not an assistant. Text below is a statistical "
    "continuation of the prompt. It may be ungrammatical, and it is never a source of facts."
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate with a SIR-Nano checkpoint (CPU-friendly).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="भारत")
    ap.add_argument("--prompt-file", help="read the prompt from a file (useful for multi-line text)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (deterministic)")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None, help="only used when --temperature > 0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", help="write the full result payload here")
    ap.add_argument("--raw", action="store_true", help="print only the generated text (no banner)")
    ap.add_argument("--repeat", type=int, default=1, help="run N times to eyeball determinism")
    args = ap.parse_args(argv)

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    model, tok, state = load_for_inference(args.checkpoint, device=args.device)
    prov = (state.get("extra") or {}).get("provisional")
    settings = GenSettings(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_ids=(int(model.cfg.eos_token_id),),
        pad_id=int(model.cfg.pad_token_id),
        seed=args.seed,
    )

    results = []
    for i in range(max(1, args.repeat)):
        r = generate(model, tok, prompt, settings)
        results.append(r)
        if not args.raw:
            if i == 0:
                print(BANNER, file=sys.stderr)
                print(f"[ckpt step={state.get('step')} val_loss={state.get('validation_loss')} provisional={prov}]", file=sys.stderr)
            print("\n--- output ---")
            print(r["text"])
            print(
                f"[{r['prompt_tokens']}+{r['generated_tokens']} tok, {r['seconds']}s, {r['tokens_per_sec']} tok/s, "
                f"degenerate={r['degenerate']}, truncated={r['truncated']}]",
                file=sys.stderr,
            )
    if args.repeat > 1:
        same = len({r["text"] for r in results}) == 1
        print(f"determinism over {args.repeat} runs: {'identical' if same else 'DIVERGED'}", file=sys.stderr)
        results.append({"deterministic_across_repeats": same})
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"checkpoint": args.checkpoint, "results": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.raw:
        print(results[0]["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
