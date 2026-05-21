# WP-CORE-5 — Parallel Scout rate-limit slot reservation

> **STATUS: ABANDONED at v1 (2026-05-21).** Codex xhigh adversarial review surfaced three CRITICALs that invalidated the spec's framing and red-phase design:
>
> 1. **CRITICAL-1** — `extract_domain_sentences` (the `ThreadPoolExecutor`-based parallel Scout the spec targeted) is **dead from production**. `analyze_document.scout_fn` at `core/architect.py:757-774` calls only `section_aware_chunks()` — no LLM, no `_wait_for_rate_limit`. The F-11 "parallel Scout race" symptom is dormant in current production code. Confirmed by audit observation 3618 (2026-05-21 10:11) from the WP-CORE-4 close-lookup sweep.
> 2. **CRITICAL-2** — The reservation pattern paces `_wait_for_rate_limit()` *returns*, not `client.chat()` *send-time*. The actual wire-level microslip lives in the post-return / pre-send window (thread scheduling + Python-side work between `with self._rate_limit_lock:` exit and `self.client.chat(...)` invocation at lines 311/411/597). A correct primitive must gate `client.chat()` *entry*, not function return.
> 3. **CRITICAL-3** — T-RATE-3 and T-RATE-4 (parallel multi-worker assertion of wall-clock gaps) would pass against the current implementation. The current lock-held `time.sleep()` already serializes returns and totals. RED tests as drafted do not red-signal F-11.
>
> Plus 5 WARN + 3 NITS + 3 OQ. Full review preserved in `.planning/pipeline_audit/decision_log.md` entry `D-CODEX-REVIEW-WP-CORE-5`.
>
> **User decision (2026-05-21 ~11:08 GMT+3):** drop WP-CORE-5 entirely; pivot iteration 4 to F-14 (`SynthesizerEmptyModelError` pipeline escape). F-11 stays OPEN with status note "dormant — parallel Scout path dead from production; reopen when `extract_domain_sentences` is rewired into `analyze_document.scout_fn` or `section_aware_chunks` gains an LLM call." Primitive correctness (the wire-level microslip) is a real-but-defense-in-depth concern; sequential Architect + Specialist calls slip 1-5 ms per gap which is within 6 s buffer in practice.
>
> **Successor WP:** WP-CORE-5b — `docs/superpowers/specs/2026-05-21-wp-core-5b-synthesizer-empty-model-policy-design.md` (F-14).
>
> Body below preserved verbatim for audit trail. Do not act on its red-phase or production-code change list.
>
> ---

**Date:** 2026-05-21
**Owner:** Baran (autonomous pipeline-hardening loop, iteration 4)
**Status:** DRAFT v1 — pending Codex xhigh adversarial review
**Parent finding:** `.planning/pipeline_audit/findings/architect.md` finding **F-11** (MAJOR)
**Loop:** Domain Pipeline Hardening Loop (fourth WP; baseline 332 confirmed at HEAD `2b8602f`)
**Sibling iterations:**
- Iteration 1 — WP-CORE-2 shipped at `25e6880` (reference-heading truncation)
- Iteration 2 — WP-CORE-3 shipped at `daefeb0` (empty-input contract)
- Iteration 3 — WP-CORE-4 shipped at `02e0fe9` (`IntermediateSaveError` + `srs_path` propagation)

---

## Motivation

`core/architect.py:_wait_for_rate_limit` (lines 145-155) is the sole rate-limit primitive across all three production LLM call sites — Scout chunk extraction (line 310), Architect identify_contexts (line 410), Specialist per-context analyze (line 596). Free-tier Gemini enforces a hard 6 s minimum delay between requests, and `DEFAULT_MIN_DELAY_SECONDS = 6.0` codifies the invariant.

Current implementation:

```python
def _wait_for_rate_limit(self):
    """Enforce minimum delay between API requests. Thread-safe."""
    with self._rate_limit_lock:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed
            print(f"  ⏳ Rate limiting... waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        self.request_count += 1
        print(f"  📡 API Request #{self.request_count}")
```

### The race (F-11)

The lock holds `time.sleep()` so workers serialize. `self.last_request_time` is timestamped after the sleep, just before lock release. The actual `client.chat()` call happens **outside** the lock (line 311 / 411 / 597). Two sources of jitter exist between unlock and request-send:

1. **Thread scheduling delay** — between `with` block exit and the next opcode of the caller, the OS may park the thread. Under load: 1–50 ms.
2. **Downstream Python work** before `client.chat()` — argument construction, attribute access, `LLMResponseAdapter` init. Bounded but observable.

Under `scout_max_workers > 1`, the worst case:

- W1 acquires lock at T = 0, sleeps 0 (first call), sets `last_request_time = T0`, releases at T = 0 + ε_lock.
- W1 emits HTTP request at T = 0 + ε_lock + ε_caller ≈ T = 0.05 s.
- W2 acquires lock at T = 0 + δ (lock contention small).
- W2 computes `elapsed = δ`, sleeps `min_delay - δ ≈ 6 s`, sets `last_request_time = T6`, releases.
- W2 emits HTTP request at T = 6 + ε_caller ≈ T = 6.05 s.

Apparent gap on the wire: **6.0 s** (correct). But:

- W3 acquires lock at T = 6 + δ' (right after W2 set `last_request_time = T6` and released; W2's HTTP request hasn't fired yet — W3 wins the lock race against W2's `client.chat()` invocation).
- W3 computes `elapsed = δ'`, sleeps `≈ 6 s`, sets `last_request_time = T12`, releases.
- W3 emits HTTP request at T = 12 + ε_caller ≈ T = 12.05 s.

Apparent wire gap W2 → W3: **6.0 s** (correct). So far OK.

**The actual slip arises when the timestamp drifts from the wire**: `last_request_time` is set at lock-release, not at request-send. If W2 is preempted between unlock and `client.chat()` — say 0.5 s of stop-the-world — its HTTP request goes out at T6.5, not T6.05. But W3 already saw `last_request_time = T6` and reserved its slot for T12.0 (not T12.5). Wire-level gap W2 → W3 = **5.5 s** — below the 6 s contract.

**Cumulative**: with N workers and per-worker jitter j, the worst-case wire-level gap on the (N-1)-th transition is `min_delay - j`. Over a long Scout pass with 30 chunks at 4 workers, accumulated slip ≈ `30 × j` — for j = 0.1 s that's 3 s of effective quota loss; for j = 0.5 s that's 15 s. On the free-tier 6 s contract, even a single missed gap triggers a quota-exhaustion 429.

Codex F-11 description: *"creating a microslip of 0.5s of combined request throughput. With many workers, these slips compound."*

### Why fix now

- **Highest-impact remaining OPEN MAJOR.** Codex pre-identified the mechanism.
- **Concurrency primitive is load-bearing**: future fixes for F-14 (Synthesizer empty escape) and F-21 (vacuous D1 verifier pass) will coexist with parallel Scout; locking the rate-limit primitive first reduces rework.
- **User-visible cost**: free-tier 429 errors are the dominant pipeline failure mode users see today. Pro-tier Gemini doesn't hit this, but the EMSE submission's "free-tier reproducibility" claim depends on `min_delay=6 s` actually holding.

### Non-goals

- **No change to default `scout_max_workers = 1`.** Sequential remains the safe default; we are not auto-enabling parallel Scout.
- **No change to `min_delay` default of 6.0 s** or `DDD_MIN_DELAY_SECONDS` env semantics.
- **No change to the 5-retry quota-error backoff** (`_is_quota_error_and_backoff`, lines 157-180). That's a separate concern (F-19 comment-code mismatch is MINOR-OPEN, deferred).
- **No removal of stdout logging** ("rate limiting…", "API Request #N"). Cosmetic; can interleave under parallel without functional impact.
- **No change to per-stage retry loops** at lines 308, 408, 590. Only the primitive changes.

---

## Design

### Pattern: atomic next-slot reservation

Replace "timestamp-after-sleep + lock-held sleep" with "advance-then-sleep-outside-lock":

```python
def _wait_for_rate_limit(self) -> None:
    """Reserve the next API-call slot atomically; sleep outside the lock.

    Invariant: consecutive successful returns from this method are at least
    ``self.min_delay`` apart in wall-clock time, regardless of caller-side
    jitter (thread scheduling, GIL contention, work done between this call
    and the actual ``client.chat()`` invocation).
    """
    with self._rate_limit_lock:
        now = time.monotonic()
        reserved = max(now, self._next_allowed_time)
        self._next_allowed_time = reserved + self.min_delay
        self.request_count += 1
        request_no = self.request_count

    sleep_time = reserved - time.monotonic()
    if sleep_time > 0:
        print(f"  ⏳ Rate limiting... waiting {sleep_time:.1f}s")
        time.sleep(sleep_time)
    print(f"  📡 API Request #{request_no}")
```

**Init change** (line 101):

```python
# was: self.last_request_time = 0
self._next_allowed_time: float = 0.0
```

### Correctness argument

Let `reserved_i` be the reservation captured by the i-th lock-holding worker.

1. **Reservation monotonicity (under lock):** `reserved_i = max(now_i, _next_allowed_time_i)`, then `_next_allowed_time_{i+1} = reserved_i + min_delay`. Therefore `reserved_{i+1} ≥ reserved_i + min_delay` by induction on i. ✓
2. **No two workers share a reserved instant** — each lock pass strictly advances `_next_allowed_time` by `min_delay`. ✓
3. **Wire-level invariant:** worker i emits its HTTP request strictly after `reserved_i` (it sleeps until then). Worker i+1 emits strictly after `reserved_{i+1} ≥ reserved_i + min_delay`. Therefore wire-level gap ≥ `min_delay` regardless of jitter between lock-release and HTTP send. ✓
4. **Initial state:** `_next_allowed_time = 0.0`, `now = monotonic_start ≫ 0`. First call: `reserved = max(now, 0) = now`. Sleep = `now - now = 0`. Request fires immediately. Subsequent: as above. ✓
5. **Sequential single-worker:** workers don't overlap; `_next_allowed_time` accumulates by `min_delay` per call; first call fires immediately, subsequent calls wait exactly `min_delay` minus their own execution time. Identical observable behavior to the old implementation when `scout_max_workers ≤ 1`. ✓

### Why `time.monotonic()` not `time.time()`

`time.time()` can step backwards under NTP correction or DST. `time.monotonic()` is non-decreasing per Python docs and is the canonical primitive for measuring intervals. Switching is safe: no test mocks `time.time()` for this primitive (verified via grep — all references to `last_request_time` are attribute pokes, not time mocks).

### Attribute rename: `last_request_time` → `_next_allowed_time`

The old name was misleading — it stored "lock-release time" (close to but not equal to "last request time"). The new attribute semantically captures "earliest instant the next request is allowed to fire", which is what the implementation actually tracks. Leading underscore signals it as private (matches `_rate_limit_lock`).

**Test impact**: 5 existing test files set `a.last_request_time = 0` directly to bypass the rate-limit gate in tests. These will be updated in the same atomic GREEN commit (no backwards-compat alias — per CLAUDE.md "no shims").

### What stays unchanged

- `self._rate_limit_lock = threading.Lock()` — still the right primitive (mutex on reservation advancement).
- `self.min_delay`, `DEFAULT_MIN_DELAY_SECONDS`, env var `DDD_MIN_DELAY_SECONDS` — untouched.
- `self.request_count` — still incremented under lock; reads safe.
- `_is_quota_error_and_backoff` — independent retry path; untouched.
- stdout "rate limiting…" and "API Request #N" messages — same content, can interleave under parallel.

---

## Red-phase tests

New file: `tests/test_scout_rate_limit_concurrency.py`. **Black-box only** — no internal-attribute pokes; tests treat `_wait_for_rate_limit` as the surface under test and measure wall-clock invariants.

| # | id | name | invariant tested |
|---|---|---|---|
| 1 | T-RATE-1 | `test_first_call_fires_immediately` | First invocation on a fresh `DomainArchitect` returns within 100 ms (no sleep on initial call). |
| 2 | T-RATE-2 | `test_sequential_two_calls_respect_min_delay` | Sequential `a._wait_for_rate_limit()` twice with `min_delay=0.3` → elapsed ≥ 0.3 s, ≤ 0.5 s. (Regression guard for single-worker.) |
| 3 | T-RATE-3 | `test_parallel_four_workers_respect_min_delay` | 4 threads invoke `_wait_for_rate_limit()` concurrently with `min_delay=0.3`. Each worker records `time.monotonic()` at the moment **just before** it returns. Sorted timestamps must have consecutive gaps ≥ 0.3 s − 10 ms tolerance. Currently expected to occasionally fail under jitter; passes deterministically after the fix. |
| 4 | T-RATE-4 | `test_parallel_eight_workers_total_walltime_ge_n_minus_1_x_delay` | 8 threads, `min_delay=0.2`, measure total wall clock from "spawn" to "last worker returns". Must be ≥ `(N-1) × min_delay − 10 ms` = 1.4 s − 10 ms. (Cumulative-slip guard.) |
| 5 | T-RATE-5 | `test_reservations_are_strictly_monotonic` | Spawn 6 workers; each records its observed-pre-sleep `now` and observed-post-sleep `now` via the primitive's behavior. Assert post-sleep observations are strictly increasing in sorted order and consecutive diffs ≥ `min_delay − 10 ms`. (Reservation atomicity guard.) |
| 6 | T-RATE-6 | `test_no_sleep_when_reservation_already_past` | If a caller is slower than `min_delay`, the next call should not sleep. Test: call once, sleep `min_delay + 0.1 s`, call again, assert elapsed ≤ 50 ms. |

**Test fixture pattern** — reuse the per-test-import + minimal-stub pattern from `tests/test_intermediate_save.py`:

```python
def _make_architect(min_delay: float = 0.3) -> "DomainArchitect":
    from core.architect import DomainArchitect
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake"}):
        with patch("core.llm.gemini.genai.Client"):
            a = DomainArchitect(min_delay=min_delay)
    return a
```

No mocking of `_wait_for_rate_limit` itself in this test file — that's the whole point.

**Tolerance**: 10 ms slack absorbs OS scheduling. Concurrency tests inherently have non-zero false-fail probability under heavy CI load; tolerance is chosen to be larger than typical Linux scheduling quantum (~1–4 ms) but small enough that the cumulative-slip bug would still trip the assertion. If CI proves flaky, tolerance can widen — but the RED→GREEN→GREEN-stable signal must be visible locally first.

**Existing test-stub updates** (same GREEN commit):

| file | line | old | new |
|---|---|---|---|
| `tests/test_intermediate_save.py` | 40 | `a.last_request_time = 0` | `a._next_allowed_time = 0.0` |
| `tests/test_architect_extraction_error.py` | 13 | `a.last_request_time = 0` | `a._next_allowed_time = 0.0` |
| `tests/test_architect_facade.py` | 14 | `arch.last_request_time = 0` | `arch._next_allowed_time = 0.0` |
| `tests/test_architect_srs_path.py` | 29 | `a.last_request_time = 0` | `a._next_allowed_time = 0.0` |
| `tests/test_specialist_per_context_loop.py` | 12 | `a.last_request_time = 0` | `a._next_allowed_time = 0.0` |

5 mechanical renames. No test logic changes (each file already patches `_wait_for_rate_limit` away, so the attribute write is just defensive zeroing).

---

## Atomic commit sequence (4 commits — matches WP-CORE-4 cadence)

| # | type | scope | summary | gate |
|---|---|---|---|---|
| 1 | `test` | `architect` | WP-CORE-5 red-phase concurrency tests for rate-limit slot reservation | Collect passes; tests run; T-RATE-3 and T-RATE-4 fail or flake under current impl. Baseline 332 → 338 collected (6 new), passing ≥ 332. |
| 2 | `fix` | `architect` | WP-CORE-5 atomic next-slot reservation for `_wait_for_rate_limit` | Touches `core/architect.py` (init + primitive rewrite) and 5 existing test stub files (mechanical rename). Baseline pytest = 338 passed, 31 deselected. |
| 3 | `chore` | `artifacts` | WP-CORE-5 dev_doc + audit state update | `development_docs/WP-CORE-5-…md` + `INDEX.md` ACTIVE row + `improvements_backlog.md` F-11 → SHIPPED + `CURRENT.md` + handoff. |
| 4 | `chore` | `planning` | WP-CORE-5 spec v2 + plan into git history | Land spec v2 (post-Codex) + plan doc. |

**Why split RED from GREEN**: standard TDD discipline. RED commit's diff isolates the new tests; reviewer can see them red against current `2b8602f` by checking out RED commit's parent and running just the new file. GREEN commit then shows the minimal production change that turns them green. Matches WP-CORE-3/4 pattern.

**Why fold test stub renames into GREEN, not a separate cleanup commit**: the rename is structurally required by the GREEN code change — the new attribute name doesn't exist before GREEN, so the renames have to land together. Not "unrelated cleanup".

---

## Risks

1. **Concurrency flakiness in CI.** Risk: T-RATE-3/4 occasionally fail on overloaded CI hardware. Mitigation: 10 ms tolerance on per-gap assertions; total-walltime test uses cumulative bound which is much looser; tests use small `min_delay=0.2`/`0.3` s so total runtime stays under 3 s per test. If a specific test proves flaky over a 10-run loop locally, widen its tolerance to 25 ms. **Do not silence with `@pytest.mark.flaky`** — would mask future regressions.
2. **Time-source mismatch.** Risk: code uses `time.monotonic()` while other architect code uses `time.time()` (e.g., the `_is_quota_error_and_backoff` 's `time.sleep(wait_time)` and the `run_timestamp` string). Mitigation: `time.monotonic()` is local to `_wait_for_rate_limit` — the new attribute `_next_allowed_time` is a monotonic-domain quantity by definition. No callsites read it; only the primitive itself does. Other `time.time()` uses are unaffected.
3. **Test stub rename misses a file.** Risk: a 6th test file references `last_request_time` and I missed it. Mitigation: grep verified the list is exhaustive at this commit (5 files). Will re-grep before GREEN commit.
4. **Behavior change for sequential users.** Risk: existing single-worker users see a behavior shift. Mitigation: §"Correctness argument" point 5 — single-worker sequential observable behavior is identical. The reservation pattern is a strict generalization of the old timestamp pattern (collapses to same wait when N=1).
5. **Print order interleaving.** Risk: parallel workers' "rate limiting…" + "API Request #N" interleave. Mitigation: cosmetic-only. `request_no` is captured under lock so the count is correct; print order in stdout is best-effort. Not a regression vs current code which already prints inside the lock (interleaving prevented but at the cost of lock-held sleep — the very bug being fixed).
6. **Monotonic clock + `time.sleep` interaction.** Risk: `time.sleep(sleep_time)` uses OS sleep which is wall-clock-driven on some platforms (Linux uses `CLOCK_MONOTONIC` via `clock_nanosleep(2)`, macOS via `mach_wait_until`). Mitigation: well-documented Python behavior; `time.sleep` is monotonic-safe on all supported platforms (CPython 3.12+).
7. **First-call regression on a long-lived architect instance.** Risk: if a `DomainArchitect` is reused across multiple `analyze_document` calls (which it is — see `main.py` lifespan), `_next_allowed_time` carries over. First call of the second invocation might wait. Verification: this is the CORRECT behavior — if the previous `analyze_document` made an API call 3 s ago, the next one should wait `min_delay − 3 s` before firing. No regression; old code did the same via `last_request_time`.

---

## Open questions for Codex xhigh

1. **OQ1 — Time source.** `time.monotonic()` vs `time.time()`. Is monotonic the right call? Are there CI/test patterns in the repo that mock `time.time()` for rate-limit assertions that would break?
2. **OQ2 — Tolerance for concurrency tests.** 10 ms slack on per-gap, but on a Linux CI runner with `nice` contention this might tighten. Alternative: drop per-gap assertion, keep only total-walltime + reservation-monotonicity. Tradeoff: coverage breadth vs. flake risk. Which weighting?
4. **OQ3 — Should `_next_allowed_time` initialize to `time.monotonic()` rather than `0.0`?** With `0.0`, the first call's `max(now, 0.0) = now`, which works because `monotonic()` returns a large positive number. But it relies on `monotonic()` start being > 0. Per Python docs, `monotonic()` may start at 0 on some platforms — would `0.0` ever break? (No, because the `max` still works — but worth verifying.)
5. **OQ4 — Should we surface `_next_allowed_time` as a public method `next_allowed()` for observability** (e.g., return `max(0, _next_allowed_time - now)` so callers can log expected wait)? Tradeoff: extra surface vs. test introspection. Recommendation: don't add — YAGNI; tests measure wall-clock directly.
6. **OQ5 — Atomic GREEN commit folds 5 test-stub renames** with the production change. Codex's prior WP-CORE-4 review preferred lean GREEN commits. Is folding renames OK because they're structurally coupled, or should renames go in a separate `test:` commit ahead of GREEN to keep diffs surgical?
7. **OQ6 — Logging of "skipped" rate-limit waits.** Current code prints `"  ⏳ Rate limiting... waiting {sleep_time:.1f}s"` only when there IS a wait. New impl preserves this (only prints when `sleep_time > 0`). For parallel debugging, would it help to also log "no wait — slot was already past" when sleep_time ≤ 0? Tradeoff: log volume. Recommendation: don't add.
8. **OQ7 — Could parallel Scout's `ex.map` ordering interact with the reservation system to create starvation?** `ThreadPoolExecutor.map` distributes work in submission order but workers may finish out-of-order. The reservation system is FIFO-by-lock-acquisition, not FIFO-by-submission. Under skew (one worker keeps grabbing the lock), is there a starvation risk? My analysis: no — `threading.Lock` in CPython is fair-ish under contention and reservations are distinct per acquire. But worth a second look.

---

## Pre-mortem (what could go wrong post-merge)

1. **A reviewer reads the GREEN diff and thinks the rename is gratuitous.** Mitigation: spec §Design explains the rename rationale; commit message refs spec.
2. **A future contributor mocks `time.monotonic()` in a test and breaks the primitive.** Mitigation: doc comment in `_wait_for_rate_limit` notes the monotonic-clock assumption.
3. **Pro-tier user with `DDD_MIN_DELAY_SECONDS=0.0`.** Spec: `min_delay=0` → `_next_allowed_time` advances by 0 per call → `reserved = max(now, last)` → `sleep_time = reserved - now ≤ 0` → no sleep. Correct degenerate behavior (no rate limit). Test T-RATE-1 covers this implicitly.
4. **Test order dependency.** If T-RATE-3 runs after T-RATE-4, the architect instance has accumulated reservations. Mitigation: each test uses a fresh `_make_architect()`. Verified by spec §Red-phase tests fixture pattern.
5. **EMSE methodology paper claim** that parallel Scout maintains the 6 s contract. After this fix, the claim is empirically defensible. If the paper already made the claim, this fix moves "aspirational" → "verified". Worth noting in `development_docs/WP-CORE-5-…md` §empirical results.

---

## Cross-references

- Finding: `.planning/pipeline_audit/findings/architect.md` §F-11
- Backlog: `.planning/pipeline_audit/improvements_backlog.md` row F-11 (will move to SHIPPED)
- Existing concurrency-test pattern: `tests/test_token_tracker_concurrency.py`
- AGENTS.md "Concurrency: thread-safety invariants must hold under arbitrarily-interleaved execution"
- CLAUDE.md "Rate limiting is real. `DomainArchitect.min_delay` defaults to 6 s (free-tier safe)."
- Sibling spec (style/cadence): `docs/superpowers/specs/2026-05-21-wp-core-4-intermediate-save-observability-design.md`
