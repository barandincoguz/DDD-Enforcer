# WP-CORE-14 — Remove synthetic context descriptions

**Status:** SHIPPED 2026-05-23
**Commits:** RED `178f20f` → GREEN `37dbc3a`
**Parent finding:** F-18 (MINOR) — SHIPPED.

## TL;DR

Two layers shipped `f"{name} context"` synthetic placeholder: architect_fn closures (intermediate JSON) and merge.py fallback (final DomainModel). WP-CORE-14 removes both; description flows through empty when LLM-derived value absent. Schema permits empty default; downstream enrich step populates.

Baseline 403 → 404 (+1 test).

## Key decisions

- **Empty over synthetic**: honest signal that description awaits enrichment vs misleading placeholder.
- **Schema relaxation**: `BoundedContext.description = Field(default="")` permits empty post-merge.
- **No new exception**: enrich step still LLM-populates; if skipped (test fixture) empty is acceptable.

## Cross-references

- WP-CORE-14 NEW invariant: Architect/Synthesizer code MUST NOT emit synthetic `f"{name} context"` placeholder descriptions. Empty string is the honest "awaiting enrichment" signal.
