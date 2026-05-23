"""Pre-flight cost estimate CLI for DDD-Enforcer LLM runs.

Usage:
    python -m scripts.cost_estimate \\
        --model gemini-3.1-pro-preview \\
        --runs 5 \\
        [--prompt-tokens 2000] \\
        [--completion-tokens 500] \\
        [--per-stage]

Defaults (if --prompt-tokens / --completion-tokens are omitted):
    --prompt-tokens 2000   (rough average across WP-CORE pipeline stages)
    --completion-tokens 500

Exit codes:
    0 — success
    2 — unknown model or invalid arguments
"""

import argparse
import sys
from pathlib import Path

# Ensure extension/backend is on sys.path so 'core' and 'configs' are importable
# when the script is invoked via `python -m scripts.cost_estimate` from any CWD.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from core.cost_estimate import (  # noqa: E402
    estimate_per_stage_cost,
    estimate_run_cost,
    estimate_single_call_cost,
)
from core.llm.registry import MODELS  # noqa: E402


_DEFAULT_PROMPT_TOKENS = 2000
_DEFAULT_COMPLETION_TOKENS = 500


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts.cost_estimate",
        description=(
            "Pre-flight USD cost estimate for a DDD-Enforcer LLM run. "
            "Reads pricing from the D1 model registry (core/llm/registry.py). "
            "Does NOT make any live API calls."
        ),
    )
    p.add_argument(
        "--model",
        required=True,
        metavar="MODEL_ID",
        help=(
            f"Model ID from the D1 registry. "
            f"Known IDs: {', '.join(sorted(MODELS.keys()))}"
        ),
    )
    p.add_argument(
        "--runs",
        type=int,
        required=True,
        metavar="N",
        help="Number of runs (>0) to project cost over.",
    )
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=_DEFAULT_PROMPT_TOKENS,
        metavar="TOKENS",
        help=(
            f"Prompt tokens per call. "
            f"Default: {_DEFAULT_PROMPT_TOKENS} (rough WP-CORE pipeline average)."
        ),
    )
    p.add_argument(
        "--completion-tokens",
        type=int,
        default=_DEFAULT_COMPLETION_TOKENS,
        metavar="TOKENS",
        help=(
            f"Completion tokens per call. "
            f"Default: {_DEFAULT_COMPLETION_TOKENS} (rough WP-CORE pipeline average)."
        ),
    )
    p.add_argument(
        "--per-stage",
        action="store_true",
        default=False,
        help=(
            "Also print a per-stage breakdown using configs.models.stage_config "
            "for each stage registered in STAGE_TO_GROUP."
        ),
    )
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # Validate model
    if args.model not in MODELS:
        print(f"[ERROR] unknown model: {args.model}", file=sys.stderr)
        return 2

    # Validate runs
    if args.runs <= 0:
        print(f"[ERROR] --runs must be > 0, got {args.runs}", file=sys.stderr)
        return 2

    # Validate token counts
    if args.prompt_tokens < 0 or args.completion_tokens < 0:
        print("[ERROR] --prompt-tokens and --completion-tokens must be >= 0", file=sys.stderr)
        return 2

    try:
        result = estimate_run_cost(
            args.model,
            args.prompt_tokens,
            args.completion_tokens,
            args.runs,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    per_call = result["per_call"]
    total = result["total"]
    runs = result["runs"]

    print(
        f"[cost-estimate] model={args.model}, runs={runs}\n"
        f"                prompt_tokens/call={args.prompt_tokens}, "
        f"completion_tokens/call={args.completion_tokens}\n"
        f"                cost/call=${per_call:.5f} USD\n"
        f"                total cost=${total:.5f} USD ({runs} runs)"
    )

    if args.per_stage:
        try:
            stage_entries = estimate_per_stage_cost(
                args.prompt_tokens, args.completion_tokens, args.runs
            )
        except ValueError as exc:
            print(f"[ERROR] per-stage cost estimation failed: {exc}", file=sys.stderr)
            return 2

        print("\n[cost-estimate] per-stage breakdown:")
        for entry in stage_entries:
            stage = entry["stage"]
            if stage == "__total__":
                print(f"  {'TOTAL':>15s}   cost/call=N/A                total=${entry['total']:.5f} USD")
            else:
                print(
                    f"  {stage:>15s}   model={entry['model_id']:<30s}  "
                    f"cost/call=${entry['per_call']:.5f} USD   "
                    f"total=${entry['total']:.5f} USD"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
