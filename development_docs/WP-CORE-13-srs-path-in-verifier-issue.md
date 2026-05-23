# WP-CORE-13 — srs_path in VerifierIssue

**Status:** SHIPPED 2026-05-23
**Commits:** RED `5675207` → GREEN `29e3ab7` → DOC `{this}` → PLANNING `{pending}`
**Spec:** `docs/superpowers/specs/2026-05-23-wp-core-13-srs-path-in-verifier-issue-design.md`
**Parent finding:** F-24 (MINOR; NEW from WP-CORE-7 OQ-6 unlocked post-F-22) — SHIPPED.

## TL;DR

Closes WP-CORE-6 A6-srs-path OQ-1 deferred follow-up. Optional `srs_path: Optional[str] = None` added to BOTH `VerifierIssue` classes (legacy dataclass + contract Pydantic). `_to_contract_issue` adapter propagates. `check_d1_supporting_sentence_ids_subset` accepts optional kwarg and threads into emitted issues. Other checks (D2-D5, D6-D8, S1) opt in as updated in future WPs.

Baseline 397 → 403 (+6 tests).

## Key decisions

- **Optional with default None** — backwards-compatible; all 13 existing call sites unchanged.
- **Two-class symmetric widening** — legacy dataclass + contract Pydantic both updated.
- **Adapter threads srs_path** — `_to_contract_issue` propagates so issue-level provenance survives the boundary.
- **D1-only opt-in for v1** — smallest correct change per AGENTS.md. Other checks (D2-D5, D6-D8, S1) get srs_path when updated in future WPs.

## Cross-references

- **Predecessor**: `[[WP-CORE-7-refiner-stage-aware]]` — F-22 SHIPPED triggered A6-srs-path follow-up.
- **WP-CORE-13 NEW invariant**: any VerifierIssue (legacy or contract) accepts optional `srs_path`; `_to_contract_issue` adapter MUST propagate.
- **Sibling**: completes the srs_path threading sweep across the orchestration-layer error taxonomy (IntermediateSaveError + SynthesizerEmptyModelError + ArchitectGroundingError + now VerifierIssue).
