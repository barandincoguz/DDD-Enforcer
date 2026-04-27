# WP-10: Bibliography + Placeholder Cleanup

**Owner:** Baran
**Depends-on:** [Phase 1.2 literature shortlist (`02-literature.md`)]
**Effort:** M
**Status:** TODO
**Addresses instructor feedback:** [Hoca-3] (primary)

## Goal

Convert paper.tex's inline `\begin{thebibliography}` (lines 924–1017) to a Springer-style external `refs.bib` BibTeX file. Resolve all 5 in-text `\placeholder{REFERENCE}` blocks (lines 146, 160, 173, 182, 215). Replace 5 placeholder bibitems (`li2024llm_bugs`, `nam2024llm_comprehension`, `dddllmindustry2024`, `automatingddd2024` ID forensics, `llmcodequality2026` author list) with verified references from `02-literature.md` shortlist. Fix the impossible arXiv ID `2603.26244`. Add 8+ new high-quality 2022–2026 refs covering DDD adoption, architecture conformance, microservice anti-patterns, LLM-for-architecture, multi-agent SE, LLM-as-Judge.

## Acceptance criteria

- [ ] `LaTeX_DL_468198_240419/refs.bib` exists with all entries.
- [ ] `paper.tex` switched from `\begin{thebibliography}` to `\bibliographystyle{spbasic}` + `\bibliography{refs}`.
- [ ] **All 🟢 entries** in `02-literature.md` verified against publisher / DOI / arXiv (Baran clicks each link; resolution confirmed).
- [ ] **All 🟡 entries** verified or downgraded; no 🟡 ships without verification.
- [ ] **All 🔴 entries** replaced with verified ones, or deleted with prose adjusted at every cite-site.
- [ ] `automatingddd2024` arXiv ID resolved: real paper found OR citation deleted with §2.5/§9.2 prose adjusted (`paper.tex` lines 91, 198, 224, 789).
- [ ] `dincoguz2025ddd_conference` venue + proceedings filled (Baran has provenance — UBMK 2025).
- [ ] `llmcodequality2026` DOI verified (`10.1007/s10664-026-10858-8`); author list filled.
- [ ] All 5 in-text `\placeholder{REFERENCE}` blocks (paper.tex lines 146, 160, 173, 182, 215) replaced with concrete cited paragraphs.
- [ ] `grep -n placeholder paper.tex` returns zero hits in the bibliography or references.
- [ ] Final compile (`latexmk -pdf paper.tex`) produces zero "missing reference" warnings.

## Implementation steps

1. Open `02-literature.md`; verify each 🟢 entry (open DOI/arXiv URL; confirm metadata). 1 hour.
2. Verify each 🟡 entry; downgrade to 🔴 if unverifiable.
3. For each 🔴 / unverified, replace with a verified alternative or delete + adjust prose.
4. **Forensic step:** search arXiv 2024 for "domain-driven design" + "LLM"; if a real paper is found, replace `automatingddd2024` ID. If not, delete the citation everywhere (lines 91, 198, 224, 789) and adjust prose to lean on `evans2024infoq` + verified industry writeup.
5. Convert thebibliography → refs.bib using a script or by hand (only ~24 entries; manual is tractable).
6. Update paper.tex preamble: `\bibliographystyle{spbasic}` + `\bibliography{refs}`.
7. Replace all 5 in-text `\placeholder{REFERENCE}` blocks with cited paragraphs (1–2 sentences each, citing 2–3 refs from the shortlist).
8. Run `latexmk -pdf paper.tex`; resolve any undefined-citation warnings.
9. Final grep: `grep -n placeholder paper.tex` should return zero in bibliography or §2.

## Outputs (file paths)

- `LaTeX_DL_468198_240419/refs.bib`
- `paper.tex` updated:
  - Preamble: `\bibliographystyle{spbasic}` + `\bibliography{refs}`
  - Lines 146, 160, 173, 182, 215: in-text placeholder paragraphs replaced
  - Lines 91, 198, 224, 789: `automatingddd2024` cite handling
  - Lines 924–1017: `\begin{thebibliography}` replaced with `\bibliography{refs}`
- Updated `02-literature.md` with verification results

## Risks & mitigations

- **Risk:** Some 🟡 entries can't be verified (authors of paper not findable). **Mitigation:** Drop them, replace with simpler alternatives (e.g., cite `hou2024llm_se_survey` for both bug detection and program comprehension if standalone refs unverifiable).
- **Risk:** `automatingddd2024` has no real replacement and the 4 in-text cites become awkward. **Mitigation:** Plan A is replacement; plan B is deletion + 1-paragraph rewrite of §2.5 + §9.2 leaning on `evans2024infoq`. Plan B is acceptable.
- **Risk:** Springer style `spbasic.bst` not found in template directory. **Mitigation:** `00-context-report.md` confirmed `spbasic.bst` IS in the LaTeX bundle (line 19 of repo dir listing). No risk.
