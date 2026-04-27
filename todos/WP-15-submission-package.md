# WP-15: Submission Package (Cover Letter, Reviewer Suggestions, Editorial Manager)

**Owner:** Baran (cover letter + reviewers); Ali (LaTeX compile dry-run)
**Depends-on:** [all other WPs done; internal pre-review passed]
**Effort:** S
**Status:** TODO
**Addresses instructor feedback:** [N/A — submission gate]

## Goal

Final submission to Springer Editorial Manager (https://www.editorialmanager.com/emse). Includes cover letter (300–500 words), 3–5 suggested reviewers (conflict-free), final compile warning-free, and a portal walkthrough.

## Acceptance criteria

- [ ] `latexmk -pdf paper.tex` produces a warning-free PDF (zero "missing reference", zero "overfull hbox > 5pt", zero "undefined citation").
- [ ] No `\placeholder{...}` remaining: `grep -n placeholder paper.tex` returns zero.
- [ ] Cover letter draft (300–500 words) addressed to EMSE editor: positioning, key findings, prior conference paper extension scope, requested associate editor (if any).
- [ ] 3–5 suggested reviewers: name, affiliation, email, 1-line conflict statement. Conflict-free (no co-authors in last 5 years, no thesis advisors, no same-institution).
- [ ] Submission folder structure follows Springer requirements (single PDF; supplementary as separate ZIP if needed).
- [ ] Replication package URL (Zenodo DOI from WP-12) in §10 Data Availability.
- [ ] ORCID for each author (verify all 3 are set up).
- [ ] Internal pre-review (1 week with 2 authors + Hoca + 1 outside reader if available) completed; final revisions integrated.
- [ ] Editorial Manager portal walkthrough done **without clicking submit** — verify all required fields populate cleanly.
- [ ] Final submission click → confirmation email captured + saved.

## Implementation steps

1. **2 weeks before submit:** Pre-review draft circulated to 2 authors + Hoca + (ideally) 1 senior outside academic. Collect feedback.
2. **1 week before submit:** Integrate feedback; final tone + grammar pass.
3. **3 days before submit:** Cover letter draft. Suggested reviewers list (cross-check with co-author database).
4. **2 days before submit:** Compile warning-free; supplementary materials sized + zipped.
5. **1 day before submit:** Portal dry-run (everything filled but not submitted).
6. **Submit day:** Click submit; capture confirmation email.

## Outputs (file paths)

- Final `paper.pdf` (single warning-free PDF)
- `cover_letter.pdf`
- `suggested_reviewers.csv` (or text file for portal copy-paste)
- `submission_log.md` (date, EM submission ID, expected first-decision date ~24 days)

## Risks & mitigations

- **Risk:** Last-minute compile warning blocks submission. **Mitigation:** WP-15 step 4 happens 2 days early, leaving time to fix.
- **Risk:** Pre-review brings major revisions. **Mitigation:** Pre-review is scheduled with a 1-week buffer specifically for this. If revisions are extensive (>2 days work), delay submission.
- **Risk:** Suggested reviewers all conflict (same circle). **Mitigation:** Identify candidates 4 weeks before submit; widen the search to international microservice / DDD / LLM-SE community.
- **Risk:** EMSE Editorial Manager portal change requires unexpected upload. **Mitigation:** Step 5 dry-run catches this 1 day in advance.
