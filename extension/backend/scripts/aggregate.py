#!/usr/bin/env python3
"""Aggregate N-run statistics for every configuration in a runs/ directory.

Usage:
    python -m scripts.aggregate --runs-root <path> [--seed 42] [--resamples 1000]

For each (pipeline, model_id, srs_label) configuration found under <runs-root>,
the script:
  1. Discovers manifest.json + *.judge.json pairs (skips runs without a judge).
  2. Groups pairs by configuration tuple.
  3. Aggregates per-type and __aggregate__ scorelines across N runs.
  4. Writes one AggregatedConfiguration JSON to <runs-root>/_aggregated/<key>.json
     atomically (tmp + fsync + os.replace).

Output lines (one per configuration):
  [OK] <config_key>: N runs aggregated → <output_path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.aggregate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-root",
        required=True,
        type=Path,
        metavar="PATH",
        help="Root directory containing run subdirectories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for bootstrap CI (default: 42).",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    runs_root: Path = args.runs_root.resolve()
    if not runs_root.exists():
        print(f"[ERROR] runs-root does not exist: {runs_root}", file=sys.stderr)
        return 1

    from core.aggregate import (
        aggregate_configuration,
        compose_aggregate_key,
        discover_run_pairs,
        group_by_configuration,
        write_aggregated_configuration_atomic,
    )

    pairs = discover_run_pairs(runs_root)
    if not pairs:
        print(f"[WARN] No manifest+judge pairs found under {runs_root}", file=sys.stderr)
        return 0

    groups = group_by_configuration(pairs)

    exit_code = 0
    for _config_tuple, group_pairs in sorted(groups.items()):
        try:
            agg_config = aggregate_configuration(
                group_pairs,
                bootstrap_resamples=args.resamples,
                seed=args.seed,
            )
            out_path = write_aggregated_configuration_atomic(agg_config, runs_root)
            key = compose_aggregate_key(agg_config)
            print(f"[OK] {key}: {agg_config.n_runs} runs aggregated → {out_path}")
        except Exception as exc:  # noqa: BLE001
            key_str = "_".join(str(k) for k in _config_tuple)
            print(f"[ERROR] {key_str}: {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
