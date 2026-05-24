# WP-CORE-28: Extension UX Wave 1

**Owner:** Any agent (post-ownership-disestablishment, 2026-05-23)
**Depends-on:** WP-01a (provider abstraction shipped) — none of the
features below need WP-01b/c data, so this WP can ship in parallel
with the paper-data chain.
**Effort:** L (4 features × ~2-3h each, split across 4 iterations)
**Status:** SPEC READY (drafted 2026-05-24 via AskUserQuestion
brainstorming with user)

## Goal

Tighten four user-facing rough edges in the VS Code extension
(`extension/src/extension.ts`, 1581 LOC monolith) without
refactoring the file layout. Each feature is small enough to ship
in one iteration, paired with a manual F5 smoke session before the
next feature starts.

## Scope (locked via brainstorming 2026-05-24)

| Decision | Value | Rationale |
|----------|-------|-----------|
| Pain points to fix | All 4 (API key onboarding, status bar progress, validation peek, backend lifecycle) | User selected all 4 in spec-design brainstorm. |
| extension.ts refactor | NONE | "Smallest correct change". Modular split is a separate WP if pain ever justifies. |
| Backend coupling | NONE | All four features consume EXISTING `/health`, `/status`, `/generate-model-stream`, `/validate` endpoints. No new backend code. |
| Manual smoke | Per-iteration F5 | User runs Extension Development Host (F5) between iters; reports back; next iter starts. |
| Test coverage | Extend `extension/src/test/extension.test.ts` with mocks per feature | No new test infra (no @vscode/test-electron CI runner). |

## Features (ranked by iteration order — ROI + risk)

### Feature 1: API key onboarding hardening (Iteration 47)

**Why first:** Pure UX. Lowest risk. Blocks every new user's first
touch. Most isolated change (no diagnostic / SSE coupling).

**Current state (`extension.ts:apiKey` flow):**
- Falls through in order: VS Code settings → env var
  `GEMINI_API_KEY` → secret storage → user prompt.
- No pre-validation: an invalid key reveals itself only when
  `/generate-model` is called minutes later.
- No auto-migration from settings/env to secret storage.

**Add:**
- Pre-validation step: on first need, ping `/health` (or attempt a
  cheap LLM call via existing backend endpoint) and verify the key
  is accepted. On failure, show user-actionable message: invalid
  key vs rate limited vs network error.
- Auto-migration prompt: if key is found in settings or env (less
  secure), offer "Move to VS Code secret storage?" toast. User
  accepts → migrate, scrub from settings. User declines → remember
  the choice (don't re-prompt every session).
- Clear status bar feedback during validation: "Validating API
  key..." → ✅ / ❌.

**Acceptance:**
- [ ] First-time user with no key sees a single prompt (existing
      behavior) but the prompt mentions auto-migration option.
- [ ] User with key in settings sees migration offer on first
      validation success.
- [ ] User with invalid key sees "API key rejected by Gemini —
      check key and try again" within 5 seconds of attempting any
      command (not minutes).
- [ ] Migration choice is remembered across sessions.
- [ ] All 4 existing API-key-related test cases in
      `extension.test.ts` still pass.
- [ ] 3 new test cases: pre-validation success, pre-validation
      failure, migration accept flow.

### Feature 2: Backend lifecycle resilience (Iteration 48)

**Why second:** Defensive. A crashing backend affects every other
feature; reliability gain compounds.

**Current state:**
- Backend child process spawned lazily via `child_process.spawn`.
- If the process exits (crash, OOM, port collision after move),
  the extension silently loses connectivity. No restart, no toast.
- Port collision: backend tries one port, gives up if taken.

**Add:**
- `child.on('exit', ...)` handler that logs exit code + signal,
  surfaces a user dialog "DDD Enforcer backend crashed (code X).
  Restart automatically?" with Yes / No / Show logs buttons.
- Auto-restart with exponential backoff (1s, 2s, 4s, 8s, 16s).
  Give up after 5 attempts; surface persistent error.
- Port collision recovery: if `/health` ping fails to bind, scan
  next 10 ports (existing `get-port` dep). Log the new port choice
  in extension output.
- Manual restart command: existing `ddd-enforcer.restartBackend`
  command gains a reason prompt for debugging.

**Acceptance:**
- [ ] Killing the backend process via `kill -9` triggers an
      extension dialog within 2 seconds.
- [ ] User-accepted auto-restart succeeds and re-establishes
      health-check within 5 seconds.
- [ ] Port collision (backend startup fails to bind) triggers
      port scan; extension reports new port via Output panel.
- [ ] After 5 failed restart attempts the extension shows a
      persistent error and does NOT loop indefinitely.
- [ ] Manual `ddd-enforcer.restartBackend` command prompts for a
      reason string (stored in extension log for debugging).
- [ ] 3 new test cases: mock spawn + exit handler, restart
      backoff sequence, port collision scan.

### Feature 3: Multi-stage progress in status bar (Iteration 49)

**Why third:** Highest visual polish ROI. Backend SSE stream
already carries per-stage events (Scout / Architect / Specialist /
Synthesizer / Verifier / Refiner) — extension just under-uses them.

**Current state:**
- Status bar shows simple "Generating model..." text during long
  pipeline runs.
- SSE stream is consumed but only the final result is rendered.

**Add:**
- Per-stage status bar update format:
  `🔄 Specialist 2/5 (40%) ETA 2m30s`
  where Specialist is the current stage, 2/5 is per-context
  progress (existing event field), 40% is overall % across all
  stages, and ETA is computed from rolling average of past stage
  durations (start naive: use last-run total).
- Spinner icon (`$(sync~spin)`) when active; idle icon when done.
- Click status bar → opens the existing Output channel for
  pipeline logs.
- Persist last-run durations to extension global state so the
  next run has an ETA basis from cold start.

**Acceptance:**
- [ ] During a `generate-model` run, status bar updates at least
      once per stage transition.
- [ ] Percent calculation matches the documented stage weights
      (Scout 10%, Architect 15%, Specialist 50%, Verifier 5%,
      Refiner 10%, Synthesizer 10%).
- [ ] ETA appears after the first stage completes (real data)
      and updates with each subsequent stage.
- [ ] Clicking the status bar opens the DDD Enforcer Output
      channel.
- [ ] Last-run durations are persisted across VS Code restarts.
- [ ] 2 new test cases: mock SSE stream, verify status bar text
      format; persist + reload durations.

### Feature 4: Validation result peek / hover (Iteration 50)

**Why last:** Most complex. Touches VS Code HoverProvider API,
needs RAG-source data caching on the extension side, and visual
review is qualitative.

**Current state:**
- Validation produces inline diagnostics underline + a Code Action
  ("Open SRS source") that jumps to the SRS file.
- No inline preview of the SRS excerpt.

**Add:**
- VS Code `HoverProvider` registered on Python files. When the
  user hovers over a violation range, look up the cached
  `/validate` response and render a Markdown hover with:
  - The violation type + severity (already in diagnostic message).
  - The top-1 RAG source: file name, page/section, ~200 chars of
    excerpt with the matching span highlighted in **bold**.
  - "Click to open" link → existing `openSourceCommand` action.
- Cache validation response per-file in extension state (LRU,
  cap 20 files) so the hover doesn't re-fetch.
- Cache invalidation: clear entry on file save (existing
  semantic-fingerprint hook already runs).

**Acceptance:**
- [ ] Hover over a diagnostic shows a Markdown popup within
      500ms (no network round-trip — uses cache).
- [ ] Popup shows top-1 SRS excerpt with the matching span
      bolded.
- [ ] "Click to open" link jumps to the same SRS location as
      the existing Code Action.
- [ ] Editing the file invalidates the cached hover content.
- [ ] LRU cap respected: opening the 21st file evicts the
      least-recently-used entry.
- [ ] 3 new test cases: hover provider returns expected Markdown
      shape; cache LRU eviction; cache invalidation on save.

## Iteration plan

| Iter | Feature | Effort | Manual smoke | Acceptance |
|------|---------|--------|--------------|------------|
| 47 | API key onboarding | ~2h | F5: try invalid key, try migration flow | 7 criteria |
| 48 | Backend lifecycle | ~2h | F5: `kill -9` backend, watch dialog + restart | 6 criteria |
| 49 | Multi-stage progress | ~2-3h | F5: run generate-model on real SRS, watch status bar | 6 criteria |
| 50 | Validation peek | ~3h | F5: validate Python file, hover over diagnostic | 6 criteria |

**Total: ~10h across 4 iterations + 4 F5 sessions.**

## Risks

| Risk | Mitigation |
|------|-----------|
| VS Code HoverProvider API quirks (Feature 4) | Mock-test extensively before F5. Fall back to Code Action only if API too fiddly. |
| Auto-restart loop (Feature 2) | Hard cap at 5 attempts. Persistent error message. |
| SSE event format drift (Feature 3) | Use a Pydantic-equivalent shape check on extension side; log and skip unknown event shapes. |
| API-key pre-validation cost (Feature 1) | Use a cheap call (e.g., model list lookup) not a real generation; respect the rate limit. |

## Out of scope

- extension.ts modularization (refactor deferred).
- Webview surfaces (those are WP-CORE-32).
- New backend endpoints (we use existing only).
- @vscode/test-electron CI runner (deferred; pair-iter F5 is the
  manual contract for now).
- Localization (status bar text stays English; UI prompts stay
  English; TR comm continues in conversation).

## Decision artefacts (brainstorming, 2026-05-24)

User's AskUserQuestion answers, in order:

1. **Pain points** (multi-select): API key onboarding, status bar
   progress, validation peek, backend lifecycle resilience (all 4).
2. **Refactor scope**: NONE (just add features).
3. **Backend coupling**: NONE (extension-only UX).
4. **Manual smoke**: Per-iter F5 with feedback loop.

Captured because they constrain implementer freedom (e.g., a
future agent must NOT introduce a backend endpoint without
user re-consent).
