#!/usr/bin/env python3
"""Render per-RQ LaTeX tables from runs/_aggregated/.

Usage:
    python -m scripts.build_tables --runs-root <path> --rq <1|2|3|4> --output <path>
    python -m scripts.build_tables --runs-root <path> --all --output-dir <dir>

The --all form generates rq1.tex through rq4.tex in <dir>/.

Pipeline:
    load_aggregated_configs → build_rqN_table → render_table_spec → write_table_tex_atomic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.latex_tables import (
    build_rq1_table,
    build_rq2_table,
    build_rq3_table,
    build_rq4_table,
    load_aggregated_configs,
    render_table_spec,
    write_table_tex_atomic,
)

_BUILDERS = {
    1: build_rq1_table,
    2: build_rq2_table,
    3: build_rq3_table,
    4: build_rq4_table,
}


def _render_one(rq_number: int, runs_root: Path, output_path: Path) -> None:
    configs = load_aggregated_configs(runs_root)
    builder = _BUILDERS[rq_number]
    spec = builder(configs)
    rendered = render_table_spec(spec)
    write_table_tex_atomic(rendered, output_path)
    print(f"[OK] RQ{rq_number}: rendered to {output_path}")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render per-RQ LaTeX tables from runs/_aggregated/.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Root directory containing the _aggregated/ subdirectory.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--rq",
        type=int,
        choices=[1, 2, 3, 4],
        help="Render a single RQ table (1–4).",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Render all four RQ tables.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for the .tex file (required when --rq is used).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for rq1.tex…rq4.tex (required when --all is used).",
    )

    args = parser.parse_args(argv)

    if args.rq is not None:
        if args.output is None:
            parser.error("--output is required when --rq is specified.")
        _render_one(args.rq, args.runs_root, args.output)
    else:
        # --all
        if args.output_dir is None:
            parser.error("--output-dir is required when --all is specified.")
        for rq_number in [1, 2, 3, 4]:
            output_path = args.output_dir / f"rq{rq_number}.tex"
            _render_one(rq_number, args.runs_root, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
