"""LaTeX table rendering for per-RQ paper data.

WP-01b Task D deliverables:
  1. TableCell          — one rendered cell (label, mean, std, formatted string).
  2. TableRow           — one row of a per-RQ LaTeX table.
  3. TableSpec          — a complete table for one RQ.
  4. sanitize_latex_label()         — whitelist alphanumerics + `-` + `_`.
  5. escape_latex_value()           — wrap free-form strings in \\detokenize{...}.
  6. format_mean_std()              — "mean ± std" with configurable decimal places.
  7. build_rq1_table()              — per-pipeline mean P/R/F1 table.
  8. build_rq2_table()              — per-model mean P/R/F1 table.
  9. build_rq3_table()              — per-violation-type mean F1 table.
  10. build_rq4_table()             — per-configuration bootstrap CI width table.
  11. render_table_spec()           — render a TableSpec to a LaTeX tabular string.
  12. load_aggregated_configs()     — glob _aggregated/*.json → validated configs.
  13. write_table_tex_atomic()      — atomic tmp+fsync+os.replace write.

No LLM calls. Pure deterministic string templating + Pydantic v2 + stdlib.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

from core.aggregate import AggregatedConfiguration, AggregatedScoreline, compose_aggregate_key


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TableCell(BaseModel):
    """One rendered cell in a P/R/F1 mean ± std table."""

    label: str
    """Already-escaped column label (e.g. "Precision")."""
    mean: float
    std: float
    formatted: str
    """e.g. "0.823 ± 0.045" — ready to drop into a LaTeX cell."""


class TableRow(BaseModel):
    """One row of a per-RQ LaTeX table.

    row_label carries the row's identifier (pipeline name, model name, etc.)
    and cells carries the numerical columns.
    """

    row_label: str
    """Already-sanitized via the LaTeX whitelist."""
    cells: List[TableCell]


class TableSpec(BaseModel):
    """A complete table for one RQ."""

    rq: Literal["RQ1", "RQ2", "RQ3", "RQ4"]
    caption: str
    label: str
    """e.g. "tab:rq1" """
    column_headers: List[str]
    """Already-escaped LaTeX strings."""
    rows: List[TableRow]


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------


def sanitize_latex_label(value: str) -> str:
    """Whitelist alphanumerics + ``-`` + ``_`` per spec risk #3.

    Any character outside ``[A-Za-z0-9_-]`` is replaced with ``_``.
    Used for row_label, model_id, srs_label.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)


def escape_latex_value(value: str) -> str:
    r"""Wrap a user-provided string in ``\detokenize{...}`` so LaTeX special
    chars (#, %, &, $, \\, ^, _, {, }, ~) do not break the document.

    Used for the caption and column_headers.
    """
    return r"\detokenize{" + value + "}"


def format_mean_std(mean: float, std: float, *, places: int = 3) -> str:
    """Return ``f'{mean:.{places}f} ± {std:.{places}f}'``.

    The ± character is the Unicode U+00B1; pdflatex with inputenc{utf8}
    renders it directly.  Tests assert on the Unicode character, NOT $\\pm$.
    """
    return f"{mean:.{places}f} ± {std:.{places}f}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_aggregate_scoreline(config: AggregatedConfiguration) -> Optional[AggregatedScoreline]:
    """Return the ``__aggregate__`` scoreline from *config*, or None."""
    for sl in config.scorelines:
        if sl.violation_type == "__aggregate__":
            return sl
    return None


def _make_prf1_cells(
    precision_mean: float,
    precision_std: float,
    recall_mean: float,
    recall_std: float,
    f1_mean: float,
    f1_std: float,
    places: int = 3,
) -> List[TableCell]:
    """Build Precision / Recall / F1 cells."""
    return [
        TableCell(
            label="Precision",
            mean=precision_mean,
            std=precision_std,
            formatted=format_mean_std(precision_mean, precision_std, places=places),
        ),
        TableCell(
            label="Recall",
            mean=recall_mean,
            std=recall_std,
            formatted=format_mean_std(recall_mean, recall_std, places=places),
        ),
        TableCell(
            label="F1",
            mean=f1_mean,
            std=f1_std,
            formatted=format_mean_std(f1_mean, f1_std, places=places),
        ),
    ]


def _average_summaries(
    configs: List[AggregatedConfiguration],
    *,
    places: int = 3,
) -> List[TableCell]:
    """Average the __aggregate__ P/R/F1 means (and stds) across a list of configs."""
    p_means, p_stds = [], []
    r_means, r_stds = [], []
    f_means, f_stds = [], []

    for c in configs:
        agg = _get_aggregate_scoreline(c)
        if agg is None:
            continue
        p_means.append(agg.precision_summary.mean)
        p_stds.append(agg.precision_summary.std)
        r_means.append(agg.recall_summary.mean)
        r_stds.append(agg.recall_summary.std)
        f_means.append(agg.f1_summary.mean)
        f_stds.append(agg.f1_summary.std)

    def _safe_mean(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return _make_prf1_cells(
        _safe_mean(p_means), _safe_mean(p_stds),
        _safe_mean(r_means), _safe_mean(r_stds),
        _safe_mean(f_means), _safe_mean(f_stds),
        places=places,
    )


# ---------------------------------------------------------------------------
# RQ table builders
# ---------------------------------------------------------------------------


def build_rq1_table(
    configs: List[AggregatedConfiguration],
) -> TableSpec:
    """RQ1: per-pipeline mean P/R/F1.

    One row per distinct ``pipeline`` value across the input configs.
    Aggregates within a pipeline by averaging the __aggregate__ scoreline
    means across all configs in that pipeline bucket.

    Caption: 'RQ1: Per-pipeline aggregate precision / recall / F1 (mean ± std across N runs).'
    Label: 'tab:rq1'
    Columns: ['Pipeline', 'Precision', 'Recall', 'F1']
    None pipeline → stored as 'unspecified' for display.
    """
    buckets: Dict[str, List[AggregatedConfiguration]] = defaultdict(list)
    for c in configs:
        key = c.pipeline if c.pipeline is not None else "unspecified"
        buckets[key].append(c)

    rows: List[TableRow] = []
    for pipeline_label in sorted(buckets.keys()):
        bucket = buckets[pipeline_label]
        cells = _average_summaries(bucket)
        rows.append(TableRow(row_label=sanitize_latex_label(pipeline_label), cells=cells))

    return TableSpec(
        rq="RQ1",
        caption=escape_latex_value(
            "RQ1: Per-pipeline aggregate precision / recall / F1 (mean ± std across N runs)."
        ),
        label="tab:rq1",
        column_headers=[
            "Pipeline",
            escape_latex_value("Precision"),
            escape_latex_value("Recall"),
            escape_latex_value("F1"),
        ],
        rows=rows,
    )


def build_rq2_table(
    configs: List[AggregatedConfiguration],
) -> TableSpec:
    """RQ2: per-model mean P/R/F1.

    One row per distinct ``model_id`` across the input configs.

    Caption: 'RQ2: Per-model aggregate precision / recall / F1 (mean ± std).'
    Label: 'tab:rq2'
    Columns: ['Model', 'Precision', 'Recall', 'F1']
    """
    buckets: Dict[str, List[AggregatedConfiguration]] = defaultdict(list)
    for c in configs:
        buckets[c.model_id].append(c)

    rows: List[TableRow] = []
    for model_id in sorted(buckets.keys()):
        bucket = buckets[model_id]
        cells = _average_summaries(bucket)
        rows.append(TableRow(row_label=sanitize_latex_label(model_id), cells=cells))

    return TableSpec(
        rq="RQ2",
        caption=escape_latex_value(
            "RQ2: Per-model aggregate precision / recall / F1 (mean ± std)."
        ),
        label="tab:rq2",
        column_headers=[
            "Model",
            escape_latex_value("Precision"),
            escape_latex_value("Recall"),
            escape_latex_value("F1"),
        ],
        rows=rows,
    )


def build_rq3_table(
    configs: List[AggregatedConfiguration],
) -> TableSpec:
    """RQ3: per-violation-type mean F1 across all configurations.

    One row per distinct violation_type observed (excluding __aggregate__).

    Caption: 'RQ3: Per-violation-type aggregate F1 (mean ± std).'
    Label: 'tab:rq3'
    Columns: ['Violation Type', 'F1']
    """
    # Group: vtype → list of (f1_mean, f1_std) across all configs
    vtype_f1_means: Dict[str, List[float]] = defaultdict(list)
    vtype_f1_stds: Dict[str, List[float]] = defaultdict(list)

    for c in configs:
        for sl in c.scorelines:
            if sl.violation_type == "__aggregate__":
                continue
            vtype_f1_means[sl.violation_type].append(sl.f1_summary.mean)
            vtype_f1_stds[sl.violation_type].append(sl.f1_summary.std)

    rows: List[TableRow] = []
    for vtype in sorted(vtype_f1_means.keys()):
        means = vtype_f1_means[vtype]
        stds = vtype_f1_stds[vtype]
        avg_mean = sum(means) / len(means) if means else 0.0
        avg_std = sum(stds) / len(stds) if stds else 0.0
        formatted = format_mean_std(avg_mean, avg_std)
        cells = [
            TableCell(label="F1", mean=avg_mean, std=avg_std, formatted=formatted)
        ]
        rows.append(TableRow(row_label=sanitize_latex_label(vtype), cells=cells))

    return TableSpec(
        rq="RQ3",
        caption=escape_latex_value(
            "RQ3: Per-violation-type aggregate F1 (mean ± std)."
        ),
        label="tab:rq3",
        column_headers=[
            "Violation Type",
            escape_latex_value("F1"),
        ],
        rows=rows,
    )


def build_rq4_table(
    configs: List[AggregatedConfiguration],
) -> TableSpec:
    """RQ4: bootstrap CI variance per configuration.

    One row per AggregatedConfiguration.
    row_label = sanitize_latex_label(f"{pipeline}_{model_id}_{srs_label}")
    CI width = ci95_high - ci95_low of the __aggregate__ F1 summary.

    Caption: 'RQ4: Per-configuration bootstrap 95% CI width for F1 (variance source).'
    Label: 'tab:rq4'
    Columns: ['Configuration', 'F1 mean', 'CI95 width']
    """
    rows: List[TableRow] = []

    for c in sorted(configs, key=compose_aggregate_key):
        pipeline_str = c.pipeline if c.pipeline is not None else "pipelineNone"
        row_label = sanitize_latex_label(f"{pipeline_str}_{c.model_id}_{c.srs_label}")

        agg = _get_aggregate_scoreline(c)
        if agg is None:
            f1_mean = 0.0
            f1_std = 0.0
            ci_width = 0.0
        else:
            f1_mean = agg.f1_summary.mean
            f1_std = agg.f1_summary.std
            ci_width = agg.f1_summary.ci95_high - agg.f1_summary.ci95_low

        cells = [
            TableCell(
                label="F1 mean",
                mean=f1_mean,
                std=f1_std,
                formatted=format_mean_std(f1_mean, f1_std),
            ),
            TableCell(
                label="CI95 width",
                mean=ci_width,
                std=0.0,
                formatted=f"{ci_width:.3f}",
            ),
        ]
        rows.append(TableRow(row_label=row_label, cells=cells))

    return TableSpec(
        rq="RQ4",
        caption=escape_latex_value(
            "RQ4: Per-configuration bootstrap 95% CI width for F1 (variance source)."
        ),
        label="tab:rq4",
        column_headers=[
            "Configuration",
            escape_latex_value("F1 mean"),
            escape_latex_value("CI95 width"),
        ],
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_table_spec(spec: TableSpec) -> str:
    r"""Render a TableSpec into a LaTeX ``tabular`` body string.

    Output format (exact for testability):

    \begin{table}[t]
    \centering
    \caption{<spec.caption — already escaped>}
    \label{<spec.label>}
    \begin{tabular}{l r...r}
    \toprule
    <header row joined by ' & ', terminated by ' \\'>
    \midrule
    <data rows ...>
    \bottomrule
    \end{tabular}
    \end{table}

    Each data row: row_label wrapped in \detokenize{<value>}, then cells
    using their ``formatted`` strings verbatim, joined by ' & ', terminated
    by ' \\'.
    """
    n_cols = len(spec.column_headers)
    # First column left-aligned; remaining right-aligned
    col_spec = "l" + " r" * (n_cols - 1)

    header_row = " & ".join(spec.column_headers) + r" \\"

    data_rows: List[str] = []
    for row in spec.rows:
        cells_str = [r"\detokenize{" + row.row_label + "}"]
        cells_str.extend(cell.formatted for cell in row.cells)
        data_rows.append(" & ".join(cells_str) + r" \\")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + spec.caption + "}",
        r"\label{" + spec.label + "}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header_row,
        r"\midrule",
        *data_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def load_aggregated_configs(runs_root: Path) -> List[AggregatedConfiguration]:
    """Glob ``<runs_root>/_aggregated/*.json`` and validate each as an
    AggregatedConfiguration.

    Skips files that fail validation with a ``[WARN]`` log line to stderr.
    Does NOT raise — the renderer should be resilient to one bad file.
    Returns a sorted list (by compose_aggregate_key).
    """
    agg_dir = runs_root / "_aggregated"
    if not agg_dir.exists():
        return []

    configs: List[AggregatedConfiguration] = []
    for path in sorted(agg_dir.glob("*.json")):
        if path.name.endswith(".tmp"):
            continue
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            config = AggregatedConfiguration.model_validate(data)
            configs.append(config)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[WARN] load_aggregated_configs: skipping {path.name}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    return sorted(configs, key=compose_aggregate_key)


def write_table_tex_atomic(content: str, output_path: Path) -> None:
    """Atomic write: delegates to ``core.io_atomic.write_text_atomic``."""
    from core.io_atomic import write_text_atomic

    write_text_atomic(output_path, content)
