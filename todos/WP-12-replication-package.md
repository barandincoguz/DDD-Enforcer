# WP-12: Replication Package (Zenodo DOI + GitHub Release)

**Owner:** Ali
**Depends-on:** [all WP-03..06 done]
**Effort:** S
**Status:** TODO
**Addresses instructor feedback:** [EMSE Open Science Initiative — required for 2024+ submissions]

## Goal

Package every artifact a future researcher needs to re-run the study, mint a Zenodo DOI for permanent reference, and reference it in §10 Data Availability (line 906–908). EMSE Open Science Initiative requires this; reviewers will check.

## Acceptance criteria

- [ ] Zenodo DOI minted (e.g., `10.5281/zenodo.NNNNN`); DOI resolves to a permanent landing page.
- [ ] Replication package zip / tarball (≤500 MB ideal; if larger, split or use Zenodo's large-file path) containing:
  - All `prompts/` (Scout, Architect, Specialist, Synthesizer, Judge rubric)
  - All `configs/` (`scope.yaml`, `model.yaml`, `pricing.yaml`)
  - All `subjects/` (D1, D2, D3 SRSes + codebases, with license notes)
  - All `seeds/` (RQ4 manifests)
  - All raw `runs/` (RQ1, RQ2, RQ3, RQ4, optional RQ5) — or a representative subset if size-constrained
  - All `judge_verdicts/`
  - Anonymized `audit_overrides.csv`
  - `REPLICATION.md` with step-by-step reproduction instructions
- [ ] GitHub release tag (e.g., `v1.0-emse-submission`) with the same content.
- [ ] §10 Data Availability (paper.tex line 907–908) updated: GitHub URL + Zenodo DOI URL + 1-paragraph contents summary.
- [ ] License: code under MIT or Apache 2.0; data under CC-BY-4.0 or similar.

## Implementation steps

1. Decide license (recommend MIT for code, CC-BY-4.0 for SRSes/run data — confirm with TEDU IP policy).
2. Write `REPLICATION.md`: step-by-step "to reproduce Table 6, run `make rq1` after `cp configs/scope.yaml.template configs/scope.yaml && export GEMINI_API_KEY=..."`.
3. Reserve Zenodo DOI (Zenodo allows DOI reservation before file upload).
4. Tag GitHub release `v1.0-emse-submission`.
5. Upload tarball to Zenodo; trigger DOI minting.
6. Update paper.tex §10 Data Availability with both URLs.
7. Smoke test: have Baran (or external grad student) follow `REPLICATION.md` from a clean clone; verify the smoke target works.

## Outputs (file paths)

- `REPLICATION.md` (in repo root or `replication_package/`)
- `replication_package/` directory layout (subfolders for each category)
- Zenodo entry (external)
- GitHub release tag
- `paper.tex` §10 Data Availability updated

## Risks & mitigations

- **Risk:** Replication package > 1 GB; Zenodo upload painful. **Mitigation:** Compress + split; alternatively, host raw runs on the GitHub release attachment (also has DOI), keep Zenodo for the metadata + key configs.
- **Risk:** SRS license restrictions block public release. **Mitigation:** Authored SRSes are unrestricted; for any third-party SRS, store only the bibliographic citation in the package.
- **Risk:** Replication target stale (e.g., `pip install` fails after a Python version bump). **Mitigation:** Pin all dependencies in `requirements.txt`; include `python-version.txt` matching the version used.
