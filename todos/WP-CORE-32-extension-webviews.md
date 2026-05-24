# WP-CORE-32: Extension Webviews — PaperRunManifest Viewer

**Owner:** Any agent (post-ownership-disestablishment, 2026-05-23)
**Depends-on:** WP-01b Task A (PaperRunManifest schema shipped),
WP-CORE-28 (UX wave 1 — same extension.ts surface; sequence after
to avoid concurrent monolith edits)
**Effort:** M (~3-5h for MVP scope)
**Status:** SPEC READY (drafted 2026-05-24 via AskUserQuestion
brainstorming with user)

## Goal

Surface paper-data PaperRunManifest artifacts inside the VS Code
extension as a read-only webview. Lets researchers (and demo
viewers) inspect a run's metrics, violations, and provenance
without leaving the editor or opening the JSON file manually.

## Scope (locked via brainstorming 2026-05-24)

| Decision | Value | Rationale |
|----------|-------|-----------|
| MVP surface | **Per-run PaperRunManifest viewer** (NOT violations dashboard, NOT metrics charts, NOT cost panel) | Smallest correct slice; read-only; reuses data already on disk. Other surfaces are future WPs. |
| Tech stack | **Vanilla HTML + JS** (no Vue, no React, no Microsoft webview-ui-toolkit) | Zero deps; ~200 LOC; quickest to ship and audit. State management isn't needed for read-only. |
| Data flow | **File-load** from disk (NOT SSE push, NOT HTTP pull) | Backend already writes manifests to `runs/<run_id>/manifest.json`. Webview reads via the extension's workspace fs API. No backend touch. |
| Sequence | **After WP-CORE-28** | extension.ts editing conflict avoidance; 28 stabilises the per-iter F5 disciplined flow. |

## MVP scope

A single VS Code command `ddd-enforcer.showRunManifests` opens a
WebviewPanel listing every PaperRunManifest discovered under the
workspace's `runs/` directory. The user can:

1. See a sortable / filterable **list** of runs (run_id, pipeline,
   model_id, srs_label, timestamp, total cost, total violations).
2. Click a run to open a **detail pane** showing the full manifest
   (pretty-printed JSON or a structured layout) — violations,
   metrics, provenance hashes (sha256_of_file,
   sha256_of_code_tree), token counts per stage.
3. Refresh on demand (button or auto-detect new files via
   `workspace.createFileSystemWatcher`).

**NOT in MVP:**
- Charts (deferred; would need a chart lib).
- Aggregated comparison across runs (deferred; that's the
  `aggregate.py` CLI's job; webview could surface its output later).
- Editing manifests (read-only).
- Backend communication (file-load only).

## Implementation steps

### Step 1: Extension command + WebviewPanel scaffold (~1h)

- Register `ddd-enforcer.showRunManifests` in `package.json`
  contributes.commands.
- In `extension.ts`, factor a small `openRunManifestsWebview()`
  function that:
  - Resolves the workspace `runs/` directory (use existing
    `WORKSPACE_PATH` env var as primary; fall back to
    `vscode.workspace.workspaceFolders[0].uri.fsPath`).
  - Creates a `WebviewPanel` with `enableScripts: true`,
    `localResourceRoots` scoped to the extension's media dir.
  - Loads inline HTML (no separate .html file — simpler ship).

### Step 2: List view rendering (~1h)

- Extension-side: glob `runs/**/manifest.json`, read each, parse
  as JSON, build a summary array (one entry per manifest).
- Send the array to the webview via `panel.webview.postMessage`.
- Webview-side: vanilla JS renders a `<table>` with columns
  (run_id, pipeline, model_id, srs_label, timestamp, cost,
  violation count). Headers clickable to sort.

### Step 3: Detail pane (~1h)

- Click a row → webview posts message to extension.
- Extension reads the full manifest JSON, posts back.
- Webview renders a structured layout: top metadata, then
  collapsible sections for violations, metrics, provenance.
- Use `JSON.stringify(..., null, 2)` for the raw-data fallback
  if a section isn't structured yet.

### Step 4: Refresh + watcher (~30min)

- Refresh button at the top of the list view.
- `vscode.workspace.createFileSystemWatcher("**/runs/**/manifest.json")`
  triggers re-discovery; webview list refreshes.

### Step 5: Tests + F5 smoke (~30min)

- Extension unit tests in `extension.test.ts`:
  - mock the glob + readFile, verify the message shape sent to
    webview.
  - verify the file watcher subscription is created on panel
    open and disposed on panel close.
- Manual F5: open Extension Development Host, run command,
  verify list + detail + refresh + sort.

## Acceptance criteria

- [ ] Command `ddd-enforcer.showRunManifests` appears in command
      palette and opens a WebviewPanel.
- [ ] List view shows every manifest under workspace `runs/`.
- [ ] Sorting works on all 7 columns.
- [ ] Clicking a row opens the detail pane on the same panel.
- [ ] Detail pane shows violations + metrics + provenance hashes
      readably.
- [ ] Refresh button re-reads the directory.
- [ ] File watcher auto-refreshes when a new manifest file is
      written.
- [ ] Webview-side JS does NOT make any external network calls
      (vanilla; verifiable by reading the inline HTML).
- [ ] No new backend endpoint or backend code change.
- [ ] 5 new test cases in `extension.test.ts` (glob + read,
      message shape, click→detail, refresh, watcher dispose).
- [ ] Manual F5 smoke passes.

## Risks

| Risk | Mitigation |
|------|-----------|
| Webview Content-Security-Policy issues with inline scripts | Use `webview.cspSource` and a `nonce` for the inline `<script>`; documented VS Code pattern. |
| Large manifests slow detail pane | Lazy-load detail JSON only on click (already in design). Cap pretty-print at 50KB; show "Open file" fallback for larger. |
| File watcher fires too often on backend writes | Debounce 500ms on the extension side. |
| Workspace with no `runs/` directory | Show a friendly empty state with a "Run pipeline first" hint. |

## Out of scope (future webview WPs)

- **Violations dashboard with filtering** (interactive aggregation
  across runs). Would build on this WP's list-view pattern.
- **Metrics charts** (precision/recall/F1 over time, token usage
  histogram). Requires Chart.js or equivalent.
- **Cost projection panel** (live wrap around `scripts/cost_estimate.py`).
- **Aggregated comparison view** (sister-pane that shows
  `aggregate.py` output and lets user filter to a configuration).

## Decision artefacts (brainstorming, 2026-05-24)

User's AskUserQuestion answers:

1. **MVP surface**: Per-run PaperRunManifest viewer.
2. **Tech stack**: Vanilla HTML + JS.
3. **Data flow**: File-load from disk (workspace `runs/`).
4. **Sequence**: After WP-CORE-28.

Future contributors who want to deviate from any of these (e.g.,
add Vue, add backend endpoint) must brainstorm with the user
first.
