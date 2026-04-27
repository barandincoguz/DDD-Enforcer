# WP-06: RQ4 Experiments — Synthetic Violation Recognition (Blind Injection)

**Owner:** **A and B split** — Author A (e.g., Baran) injects + writes manifest; Author B (e.g., Ali) runs blind; Author A scores. Roles are fixed in writing.
**Depends-on:** [WP-01d, WP-02 (codebases ready), WP-03, WP-04, WP-07]
**Effort:** L (corpus generation + blind protocol + 90 seeds × 5 runs)
**Status:** BLOCKED on WP-04
**Addresses instructor feedback:** [Hoca-1] (fills Table 9)

## Goal

Build a controlled-recall dataset by deliberately seeding **5 violations per type per domain × 6 types × 3 domains = 90 seeds total** into the WP-02 reference codebases. Use a **blind injection protocol** to defuse the "same-author-injects-and-evaluates" reviewer threat (R5 in `01-risks.md`, B.5 in `01-brainstorming.md`). Produce Table 9 cells + qualitative analysis of which violation types the framework recovers most reliably.

## Acceptance criteria

- [ ] `seeds/D1/manifest.yaml`, `seeds/D2/manifest.yaml`, `seeds/D3/manifest.yaml` — each with 30 entries (5 per V1–V6).
- [ ] Each manifest entry has: `id`, `type` (V1–V6), `file_edited`, `line_edited`, `edit_summary`, `rationale`, `git_commit_hash` (the commit that injected it).
- [ ] Manifests are **committed by Author A *before* Author B sees them**. Git history demonstrates the role-separation.
- [ ] Author B's runs do not have access to the manifest path during execution.
- [ ] After Author B completes runs, Author A scores: each seeded violation is matched against framework reports by `(file, type)` pair (right-file + right-type counted as detected).
- [ ] `LaTeX_DL_468198_240419/tables/rq4.tex` rendered with Seeded / Detected / Seeded-recall × V1–V6 + Overall row.
- [ ] §8 analysis paragraph (line 738): most/least reliable types, non-seeded confirmed bonuses, type-confusion analysis.
- [ ] §8 summary (line 742): recall floor framing, weakest type identified.
- [ ] §4.6 (`sec:exp_synthetic`) prose updated to explicitly describe the blind injection protocol (closes R5).

## Implementation steps

1. **Author A only:** generate reference codebases for D1, D2, D3 using a strong non-tested non-Judge LLM (e.g., Claude Opus 4.7 in instruct-only mode). Light cleanup pass.
2. **Author A only:** design seed catalogue — 5 per V1–V6 per domain. Spread across multiple files; vary edit difficulty (some obvious, some subtle).
3. Author A injects seeds; commits manifest to a separate branch `wp-06-seeds`.
4. **Lock-step gate:** Author A merges the seeded codebase into `subjects/D{1,2,3}/code-seeded/` *and* commits the manifest to `seeds/<domain>/manifest.yaml`. This is the audit trail.
5. **Author B only:** runs `make rq4` on the seeded codebases (using WP-04 winning config × 5 runs). B never reads `seeds/*/manifest.yaml` — codebase access via `subjects/D{1,2,3}/code-seeded/` only.
6. Author A scores: matches detection output against the manifest by `(file, type)`.
7. Build Table 9; write §8 analysis.
8. **Bonus signal:** if framework reports additional violations not in the manifest, Author A asks Judge whether they're real — these become a "non-seeded confirmed" sub-table in the replication package.

## Outputs (file paths)

- `subjects/D{1,2,3}/code-seeded/` (codebases with seeded violations)
- `seeds/D{1,2,3}/manifest.yaml`
- `runs/rq4/<domain>_<run>/manifest.json` (15 runs total: 3 domains × 5 runs)
- `judge_verdicts/rq4/`
- `LaTeX_DL_468198_240419/tables/rq4.tex`
- `paper.tex` §4.6 (blind injection protocol prose) + §8 analysis + summary
- `replication_package/rq4_seed_match_log.csv`

## Risks & mitigations

- **Risk:** Two-author "blindness" is leaky in a small team. **Mitigation:** Author B uses a separate working directory; Author A locks `seeds/` directory permissions. Document protocol in §4.6.
- **Risk:** Type-confusion (framework detects V1 where seed was V3) inflates apparent recall. **Mitigation:** Strict scoring: `(file, type)` match is required. Type-confusion goes in the analysis paragraph.
- **Risk:** Seeded violations are too easy or too hard, distorting recall. **Mitigation:** Mix difficulty within each type (per Step 2 above); document the mix in `seeds/README.md`.
- **Risk:** Generated codebase has spurious DDD violations introduced by the generator. **Mitigation:** Per RQ4 protocol, framework reports are Judge-audited; spurious reports go into the "non-seeded confirmed bonus" bucket and don't degrade recall.
