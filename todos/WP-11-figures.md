# WP-11: Figures + Diagrams

**Owner:** Baran (Figures 1, 4 design + caption); Ali (Figure 2 matplotlib script)
**Depends-on:** [WP-04 for Figure 2 data; WP-16 for Figure 4 specs]
**Effort:** M
**Status:** TODO
**Addresses instructor feedback:** [Hoca-4] (primary — system architecture + extension workflow figures)

## Goal

Produce 4 figures, all vector PDF, all in flat directory `figures/` (no subfolders, per Springer EMSE format requirement).

- **Figure 1**: System architecture diagram (replaces line 258 placeholder).
- **Figure 2**: RQ2 cost-accuracy Pareto frontier (replaces line 672 placeholder).
- **Figure 3**: RQ4 per-type seeded recall bar chart (NEW — current paper does not request it; recommended for clarity).
- **Figure 4**: Extension workflow sequence diagram (NEW per Hoca-4 — supports §3.4 expansion).
- **Figure 5 (optional)**: VS Code extension screenshot (replaces line 386 AUTHOR-TODO).

## Acceptance criteria

- [ ] `figures/architecture.pdf` (Figure 1): vector PDF showing SRS → Multi-Agent Pipeline (Scout→Architect→Specialist→Synthesizer) → Domain Model JSON → Code Analysis Engine (AST + LLM Validator) → Violations → Traceability Pipeline → VS Code Extension. Tool: TikZ or draw.io export.
- [ ] `figures/rq2_pareto.pdf` (Figure 2): scatter plot, log-x cost-per-validation, y F1, one point per tested model, Pareto frontier annotated. Tool: matplotlib (`scripts/figures/pareto.py`).
- [ ] `figures/rq4_seeded_recall.pdf` (Figure 3): bar chart of per-V1–V6 seeded-recall, error bars for N=5 runs.
- [ ] `figures/extension_sequence.pdf` (Figure 4): sequence diagram (developer save → extension → backend HTTP → validation → diagnostic display → "View Source" action). Tool: PlantUML or TikZ.
- [ ] `figures/extension_screenshot.png` + `.pdf` (Figure 5, optional): real screenshot, 600 dpi.
- [ ] All figures referenced via `\includegraphics{figures/architecture}` (no subfolders).
- [ ] Captions written and ≤2 lines each.
- [ ] Compile-test: `latexmk -pdf paper.tex` succeeds with all figures.

## Implementation steps

1. **Figure 1**: Sketch on paper first, then render. TikZ recommended (svjour3-friendly). Mirror the structure of §3 of paper.
2. **Figure 2**: Once WP-04 produces `runs/rq2/`, write `scripts/figures/pareto.py` to scatter `cost_usd` (x, log) vs `f1_mean` (y), one point per model with error bars. Annotate frontier (models on the convex hull).
3. **Figure 3**: Once WP-06 produces `runs/rq4/`, bar chart with `matplotlib`.
4. **Figure 4**: PlantUML or TikZ sequence diagram. Inputs come from WP-16 (extension architecture documentation).
5. **Figure 5**: Live VS Code session, capture diagnostic + side panel; trim and embed.
6. Verify all figures are vector (PDF or EPS); raster only for screenshots; embed at 600 dpi if raster.
7. Update `paper.tex` line 258 and 672 placeholders with `\includegraphics`.

## Outputs (file paths)

- `LaTeX_DL_468198_240419/figures/architecture.pdf` (and `.tex` source if TikZ)
- `LaTeX_DL_468198_240419/figures/rq2_pareto.pdf`
- `LaTeX_DL_468198_240419/figures/rq4_seeded_recall.pdf`
- `LaTeX_DL_468198_240419/figures/extension_sequence.pdf` (and `.puml` source)
- `LaTeX_DL_468198_240419/figures/extension_screenshot.{png,pdf}`
- `scripts/figures/pareto.py`, `scripts/figures/rq4_bars.py`
- `paper.tex` figure references updated

## Risks & mitigations

- **Risk:** Figure 1 architecture is over-complex; reviewer can't parse. **Mitigation:** Two-pass review: Baran sketches; Ali (fresh eyes) validates "can I describe this in 1 sentence?".
- **Risk:** Pareto frontier is degenerate (1 model dominates). **Mitigation:** Still produce the figure; degeneracy is the result. Annotate the dominator.
- **Risk:** Extension screenshot looks dated / unpolished. **Mitigation:** Use a curated example (V1 Synonym, clear violation, professional theme like Default Dark+).
