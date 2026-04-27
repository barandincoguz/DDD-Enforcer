# WP-16: Extension Architecture Documentation (Hoca-4 — Extension Workflow)

**Owner:** Baran (writing) + Ali (screenshots + extension-flow accuracy check)
**Depends-on:** [WP-11 (Figure 4 design coordination)]
**Effort:** M
**Status:** TODO
**Addresses instructor feedback:** [Hoca-4] (primary, extension half)

## Goal

Expand §3.4 "IDE Integration" from its current ~7-line footprint (paper.tex lines 380–386) to a full 1.5-page subsection covering: command flow, settings, diagnostic provider lifecycle, "View Source" action sequence, performance characteristics, and 3–5 real screenshots. Hoca-4 explicitly flags this; reviewer "developer experience" question (B.6 sub-issue) is partially mitigated here.

## Acceptance criteria

- [ ] §3.4 expanded to ~1.5 pages including:
  - The 4 command palette commands (from `00-context-report.md` Q25): `Initialize Domain Model`, `Validate Current File`, `Show Status`, `Restart Backend Server` — and their typical use case.
  - Workspace settings + extension settings exposed.
  - Diagnostic provider lifecycle: file save → backend HTTP → Pydantic-validated response → `vscode.Diagnostic` rendered.
  - "View Source" action sequence (paper L385): retrieves SRS paragraph from traceability pipeline, opens side panel.
  - Performance: validation latency in seconds (typical case), where it's perceptible.
- [ ] Figure 4 (extension sequence diagram, produced by WP-11): developer save → extension dispatch → Python backend HTTP → validation → diagnostic display → "View Source" UI.
- [ ] 3–5 screenshots: in-editor diagnostic, hover tooltip, problems panel, "View Source" side panel, status-bar item. Embedded in §3.4 or as a separate Figure 5.
- [ ] All screenshots taken from a clean VS Code session with a representative SRS+codebase (e.g., D1).

## Implementation steps

1. Read existing `extension/src/extension.ts` (44 KB) to confirm command list + diagnostic flow.
2. Run the extension live (after WP-01d backend is rebooted on a clean machine); capture screenshots.
3. Sketch Figure 4 sequence (PlantUML or TikZ); coordinate with WP-11 for final rendering.
4. Write the §3.4 expansion (1.5 pages = ~600 words + diagrams).
5. Hoca review gate: share §3.4 + screenshots; iterate if review feedback requests changes.
6. Cross-check that Figure 1 (architecture, WP-11) and Figure 4 (sequence) are *consistent* — same naming for components.

## Outputs (file paths)

- `paper.tex` §3.4 expanded (lines 380–386 → ~150 lines)
- `figures/extension_sequence.pdf` (Figure 4, joint with WP-11)
- `figures/extension_*.png` (3–5 screenshots, joint with WP-11)
- `docs/extension_architecture.md` (long-form internal documentation; subset goes into paper)

## Risks & mitigations

- **Risk:** Extension's actual implementation differs from planned narrative. **Mitigation:** Read `extension/src/extension.ts` first; document what *is*, not what should be.
- **Risk:** Screenshots show internal details (paths, tokens) that shouldn't be public. **Mitigation:** Use a clean demo workspace; redact API keys; use generic SRS content for the demo file.
- **Risk:** §3.4 grows beyond 1.5 pages and exceeds Springer recommended section length. **Mitigation:** Move detailed extension internals to a §"Implementation Notes" appendix or to `docs/extension_architecture.md`.
