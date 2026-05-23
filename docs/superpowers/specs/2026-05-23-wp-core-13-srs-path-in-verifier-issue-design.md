# WP-CORE-13 — srs_path in VerifierIssue (F-24)

**Status:** SHIPPED. Closes WP-CORE-6 A6-srs-path OQ-1 deferred follow-up.

## Motivation

WP-CORE-7 spec deferred A6-srs-path as F-24 backlog with trigger "post-F-22 SHIPPED". F-22 shipped at WP-CORE-7 `ce56d99`. Trigger fired.

## Design

Add `srs_path: Optional[str] = None` to BOTH:
- `core/verifier/types.VerifierIssue` (frozen dataclass)
- `core/pipeline_contracts.VerifierIssue` (Pydantic BaseModel)

Update `_to_contract_issue` adapter at `core/architect.py:879-890` to propagate srs_path.

Update `check_d1_supporting_sentence_ids_subset` to accept optional `srs_path` kwarg and thread into emitted issues.

Other checks (D2-D5, D6-D8, S1) intentionally left out of v1 — opt in as updated in future WPs. Default `None` preserves back-compat for all 13 existing call sites.

## Tests

T-SRS-1/1b/2/2b/3/4 in `tests/test_verifier_issue_srs_path.py`.

## Empirical

Baseline 397 → 403 (+6 tests, zero regression).
