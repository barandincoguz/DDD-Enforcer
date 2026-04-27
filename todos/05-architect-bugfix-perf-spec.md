# Spec — `core/architect.py` Bug Fixes + Performance (B-scope)

**Date:** 2026-04-27
**Branch:** `feat/EnhancedDocumentParserModule` (existing).
**Status:** APPROVED — Sections 1 + 2 confirmed by Baran.
**Scope:** Surgical bug fixes (B1–B5) and bounded performance wins (P1–P4) on `extension/backend/core/architect.py`. No file split; no architectural rewrite.
**Out of scope:** Q1 file split (879 → modules), Q2 stage dedup helper, A1–A4 architectural redesign, P5 SHA cache, logging migration. Reserved for a separate scope (option C/D from triage).

---

## 1. Goal

Make `core/architect.py` correct (close 5 bugs including `architect-bug-001`) and meaningfully faster (configurable rate-limit, opt-in parallel Scout chunks, head+tail truncation that preserves late-document concepts) while keeping the public API and current default behavior intact. A user who runs the existing entrypoints with no new arguments and no env vars must observe identical behavior to today, except that:

- `MAX_TOKENS` truncation no longer triggers futile retries (closes `architect-bug-001`).
- `response.text == None` (e.g., SAFETY/RECITATION block) routes through the existing parse-failure path instead of raising `AttributeError`.
- The Architect/Specialist stages preserve the document's tail when input exceeds the per-stage `max_chars` budget.

Power users gain three opt-in knobs:

- `DDD_MIN_DELAY_SECONDS` env var (default `6.0`) — drops Pro-tier wall-clock by ~6× when set to `1.0`.
- `DDD_SCOUT_MAX_WORKERS` env var (default `1`) — Scout chunks process in parallel when raised.
- Constructor args (`min_delay`, `scout_max_workers`) for programmatic override.

---

## 2. Context

**Source file:** `extension/backend/core/architect.py` (879 lines as of HEAD `03f447a`).

**Diagnostic findings** (from final cumulative review of the model-registry refactor + this brainstorm):

- Lines 235, 335, 471, 613 pass `response.text` (typed `str | None` by the Gemini SDK) into `self._parse_json_response(response_text: str)`. Pyright flagged this throughout the model-registry session.
- Lines 802–840 (`_check_response_completion`) return `False` for any non-`STOP` finish reason. The 4 stage retry loops then re-issue the same prompt; for `MAX_TOKENS` this is futile (same prompt → same truncation).
- Lines 224–225, 324–325, 460–461, 602–603 hardcode `temperature=0.05, seed=42` even though `configs/models.py` exposes `stage_config(stage).temperature` and `.seed` for each stage.
- Line 48 hardcodes `self.min_delay = 6.0`. The 4-stage pipeline accumulates ≥24 s of artificial sleep at this setting.
- Lines 290–293 (Architect) and 421–424 (Specialist) truncate input via `text[:max_chars]`, dropping the document tail. SRS documents commonly place acceptance criteria and domain events near the end; these silently disappear.
- Scout chunks (138–143) iterate sequentially; each chunk's LLM call waits the full `min_delay` before the next.
- Line 730 prints `🎯 Ready for final synthesis` but `analyze_document` does not call `synthesize`. The caller has to.
- `_handle_quota_error` (89–111) returns `int` (sleep duration or 0), and callers branch on `== 0`. Function does triple duty: detect, sleep, return sentinel.
- `_parse_json_response` is annotated `Dict[str, Any]` but actually returns a `dict`, a `list`, or a sentinel error-dict. Three call sites (258, 370, 382) check `isinstance(result, list)`.

**Existing infrastructure to leverage:**

- `configs/models.py` (`stage_config`, `STAGE_TO_GROUP`) — already exports `temperature`, `seed`, `model_id` per pipeline stage. No registry changes needed for this scope.
- `tests/test_models_registry.py` and `tests/test_token_tracker_v2.py` — established pytest patterns; new tests for this scope follow the same conventions.

---

## 3. Constraints

- **AGENTS.md compliance.** `architect.py` is already 879 lines (preexisting violation, out of B-scope). This refactor must not grow the file by more than ~50 lines net. No file split (deferred to option C scope).
- **Public API preservation.** `DomainArchitect.__init__`, `extract_domain_sentences`, `identify_contexts`, `extract_all_contexts_details`, `synthesize`, `synthesize_final_model`, `analyze_document` all keep their current signatures. Two new optional kwargs are added to `__init__` (`min_delay`, `scout_max_workers`) with `Optional[T] = None` defaults — pre-existing call sites unaffected.
- **Default behavior preservation.** With no new env vars and no new constructor kwargs, the pipeline behaves exactly as it does at HEAD `03f447a` except for the three explicit corrections noted in §1.
- **No new runtime dependencies.** `concurrent.futures.ThreadPoolExecutor` is stdlib; `threading.Lock` is stdlib; everything else already present.
- **Test discipline.** Each behavioral change ships with a focused test (3–5 new tests total). Existing 71 tests must remain green.

---

## 4. Bug Fixes (B1–B5)

### 4.1 B1 — `response.text` Optional handling

**New helper** on `DomainArchitect`:

```python
def _safe_response_text(self, response) -> str:
    """Return response.text or empty string if None.

    Gemini may return `response.text = None` when finish_reason is SAFETY,
    RECITATION, or when no candidates were produced. The existing
    _parse_json_response treats empty input as a JSON parse failure and
    routes through the retry path; surfacing None directly would raise
    AttributeError instead.
    """
    text = getattr(response, "text", None)
    return text if text is not None else ""
```

**Sites updated:** lines 235, 335, 471, 613 — replace `response.text` with `self._safe_response_text(response)`.

### 4.2 B2 — `MAX_TOKENS` accept-truncated (closes `architect-bug-001`)

**Edit `_check_response_completion` (lines 802–840):**

```python
finish_reason = response.candidates[0].finish_reason

if finish_reason == "STOP":
    return True

if finish_reason == "MAX_TOKENS":
    # Same prompt would re-truncate; accept whatever JSON arrived and
    # let _parse_json_response handle truncated/incomplete output.
    print("      💡 Hit max output tokens — accepting partial response")
    return True

# SAFETY / RECITATION / OTHER may be transient — retry can help.
print(f"      ⚠️  Response incomplete: finish_reason={finish_reason}")
if finish_reason == "SAFETY":
    print("      🛡️  Response blocked by safety filters")
elif finish_reason == "RECITATION":
    print("      📝 Response blocked due to citation/recitation")
elif finish_reason == "OTHER":
    print("      ❓ Response stopped for unknown reason")

if retry < 4:
    print(f"      🔄 Retrying ({retry + 1}/5)...")
    return False
print("      ❌ Max retries reached")
return False
```

The dead `# TODO(architect-bug-001):` comment block is **removed** in this edit. The latent `MAX_OUTPUT_TOKENS` field reference is gone.

### 4.3 B3 — `_parse_json_response` return type honesty

**Update signature + docstring (line 855):**

```python
def _parse_json_response(self, response_text: str) -> Dict[str, Any] | List[Any]:
    """Parse JSON from LLM response.

    Returns the parsed structure (dict or list — Gemini sometimes returns
    a top-level array). On parse failure returns the sentinel
    {"error": "json_parse_failed", "raw_response": <first 500 chars>}.
    Callers detect failure with:
        isinstance(result, dict) and result.get("error") == "json_parse_failed"
    """
```

No body change. Callers already handle dict/list polymorphism (lines 258, 370, 382). Pyright will start surfacing mis-uses if any exist downstream.

### 4.4 B4 — Truthful `analyze_document` log

**Replace line 730:**

```python
print(f"  🎯 Returning {len(results)} context analyses; caller invokes synthesize() next")
```

### 4.5 B5 — `_handle_quota_error` → `_is_quota_error_and_backoff`

**Rename and re-shape (lines 89–111):**

```python
def _is_quota_error_and_backoff(self, error: Exception, retry_count: int) -> bool:
    """Return True iff `error` was a quota / rate-limit error AND we slept
    for the recommended backoff duration. Return False for any other error
    (caller decides how to handle).
    """
    error_str = str(error)
    is_quota = (
        "429" in error_str
        or "quota" in error_str.lower()
        or "ResourceExhausted" in str(type(error))
    )
    if not is_quota:
        return False

    retry_match = re.search(r"retry in (\d+\.?\d*)", error_str)
    if retry_match:
        wait_time = max(float(retry_match.group(1)), 10)
    else:
        wait_time = min(15 * (2 ** retry_count), 300)

    print(f"  ⚠️  QUOTA EXCEEDED - Backing off {wait_time:.1f}s...")
    time.sleep(wait_time)
    return True
```

**Caller updates** at lines 263, 391, 514, 659:

```python
except Exception as e:
    if not self._is_quota_error_and_backoff(e, retry):
        # Existing non-quota fallback path
        print(f"      [WARN] ... error: {e}")
        if retry >= 4:
            return <existing fallback>
```

The `if X == 0:` sentinel disappears.

---

## 5. Performance (P1–P4)

### 5.1 P3 — Registry-driven `temperature` + `seed`

Each of the 4 stages reads its `StageConfig` once and uses its fields:

```python
# Scout (stage 1)
sc = stage_config("Scout")
response = self.client.models.generate_content(
    model=self.model_name,
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=sc.temperature,
        seed=sc.seed,
    ),
)
```

Same pattern applied to Architect, Specialist, Synthesizer with their respective stage names. The `sc` lookup happens at the top of each stage method (not per chunk) to avoid repeated dict lookups.

`stage_config(...)` already raises `KeyError` on unknown stage names, so the registry-validation guarantees at import time prevent the broken path.

### 5.2 P1 — Configurable `min_delay`

**Module-level constant** in `architect.py`:

```python
DEFAULT_MIN_DELAY_SECONDS = 6.0
```

**`__init__` signature gains `min_delay`:**

```python
def __init__(
    self,
    model: Optional[str] = None,
    progress_callback: ProgressCallback = None,
    min_delay: Optional[float] = None,
    scout_max_workers: Optional[int] = None,  # see P2
):
    ...
    self.min_delay = (
        min_delay
        if min_delay is not None
        else float(os.getenv("DDD_MIN_DELAY_SECONDS", DEFAULT_MIN_DELAY_SECONDS))
    )
```

The hardcoded `self.min_delay = 6.0` at line 48 is replaced. No behavior change without env var or kwarg. Setting `DDD_MIN_DELAY_SECONDS=1.0` cuts the artificial delay 6×.

### 5.3 P2 — Parallel Scout chunks

**Module imports:**

```python
import os
import threading
from concurrent.futures import ThreadPoolExecutor
```

**Module-level default (line resolves at first import):**

```python
DEFAULT_SCOUT_MAX_WORKERS = max(1, int(os.getenv("DDD_SCOUT_MAX_WORKERS", "1")))
```

**`__init__` adds:**

```python
self._rate_limit_lock = threading.Lock()
self.scout_max_workers = (
    scout_max_workers
    if scout_max_workers is not None
    else DEFAULT_SCOUT_MAX_WORKERS
)
```

**`_wait_for_rate_limit` body wrapped:**

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

**`extract_domain_sentences` chunk loop:**

```python
if self.scout_max_workers <= 1:
    # Sequential — preserves current behavior exactly.
    for i, chunk in enumerate(chunks):
        progress = int((i + 1) / len(chunks) * 100)
        print(f"  ▶️  Chunk {i + 1}/{len(chunks)} ({len(chunk):,} chars) [{progress}%]")
        self._report_progress(
            "Scout", "in_progress",
            f"Processing chunk {i + 1}/{len(chunks)}", progress
        )
        sentences = self._extract_sentences_from_chunk(chunk, i + 1, len(chunks))
        all_sentences.extend(sentences)
else:
    # Parallel — coalesce per-chunk progress to start/end of stage.
    print(f"  ⚡ Parallel mode: {self.scout_max_workers} workers")
    self._report_progress(
        "Scout", "in_progress",
        f"Processing {len(chunks)} chunks in parallel ({self.scout_max_workers} workers)",
        50,
    )
    args = [(chunk, i + 1, len(chunks)) for i, chunk in enumerate(chunks)]
    with ThreadPoolExecutor(max_workers=self.scout_max_workers) as ex:
        chunk_results = list(ex.map(
            lambda a: self._extract_sentences_from_chunk(*a),
            args,
        ))
    for r in chunk_results:
        all_sentences.extend(r)
```

`executor.map` preserves submission order, so `all_sentences` keeps deterministic ordering across runs.

`_extract_sentences_from_chunk`'s existing per-chunk retry loop is unchanged; failures within a chunk return `[]` (best-effort) and the rest of the batch continues.

**Thread-safety contract.** Each Scout worker calls `self.token_tracker.track_api_call(...)`, which mutates shared counters on the singleton `TokenTracker` instance. Under Python's GIL, `self.stats.total_api_calls += 1` is read-modify-write across two bytecodes — concurrent workers can lose updates. This task adds a `threading.Lock` around the mutation block of `TokenTracker.track_api_call` (lines 95–120 of `core/token_tracker.py`):

```python
# core/token_tracker.py — added at class level
def __init__(self) -> None:
    self.stats = TokenUsageStats()
    self.session_start = datetime.now().isoformat()
    self._lock = threading.Lock()

# track_api_call body — wrap mutation block
with self._lock:
    self.stats.total_prompt_tokens += billable_prompt
    self.stats.total_completion_tokens += completion_tokens
    # ... (all subsequent self.stats.* mutations and call_history append)
```

The lock is acquired only during the ~1 µs counter update — no impact on serial-mode performance. Existing `tests/test_token_tracker_v2.py` continues to pass without modification (the contract is preserved). One additional smoke test is added that runs `track_api_call` from 8 threads × 100 calls each and asserts `total_api_calls == 800`.

### 5.4 P4 — Head + tail truncation helper

**New module-level helper** (inside `core/architect.py`, NOT on the class):

```python
def _truncate_with_head_tail(text: str, max_chars: int, head_ratio: float = 0.6) -> str:
    """Truncate `text` to `max_chars` by keeping the head and tail and
    dropping the middle, with an explicit marker so the LLM knows the
    document was truncated.

    Default ratio: 60 % head / 40 % tail. SRS documents typically place
    intro / core concepts up front and acceptance criteria / domain
    events near the end; both should survive context-window squeezes.
    """
    if len(text) <= max_chars:
        return text
    marker = "\n\n... [middle truncated for context window] ...\n\n"
    budget = max_chars - len(marker)
    head_size = int(budget * head_ratio)
    tail_size = budget - head_size
    return f"{text[:head_size]}{marker}{text[-tail_size:]}"
```

**Sites updated:**

- Line 290–293 (`identify_contexts`): replace `text = text[:max_chars]` with
  ```python
  text = _truncate_with_head_tail(text, max_chars=50000)
  ```
- Line 421–424 (`extract_all_contexts_details`): replace `sentences_text = sentences_text[:max_chars]` with
  ```python
  sentences_text = _truncate_with_head_tail(sentences_text, max_chars=60000)
  ```

The pre-existing console message ("Truncating input: X → Y chars") stays in place — it's accurate (head+tail still totals `max_chars`).

---

## 6. New Tests

Added to `tests/test_unit.py` (or a new `tests/test_architect_helpers.py` if the file is already over the soft cap — prefer the new file to avoid further bloating `test_unit.py`).

### 6.1 `_safe_response_text`

```python
class TestArchitectSafeResponseText:
    def test_returns_text_when_present(self):
        from unittest.mock import Mock
        from core.architect import DomainArchitect
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake"}):
            with patch("core.architect.genai"):
                a = DomainArchitect()
                resp = Mock()
                resp.text = "hello"
                assert a._safe_response_text(resp) == "hello"

    def test_returns_empty_when_text_is_none(self):
        # … same fixture, with resp.text = None → returns ""

    def test_returns_empty_when_text_attr_missing(self):
        # … mock without .text attribute → returns ""
```

### 6.2 `_truncate_with_head_tail`

```python
class TestTruncateWithHeadTail:
    def test_no_truncation_when_under_budget(self):
        from core.architect import _truncate_with_head_tail
        assert _truncate_with_head_tail("short", 100) == "short"

    def test_keeps_head_and_tail(self):
        from core.architect import _truncate_with_head_tail
        text = "A" * 1000 + "B" * 1000
        result = _truncate_with_head_tail(text, max_chars=400)
        assert result.startswith("A" * 100)  # 60% × (400 - marker) head
        assert result.endswith("B" * 100)    # remainder tail
        assert "[middle truncated" in result

    def test_total_length_within_budget(self):
        from core.architect import _truncate_with_head_tail
        text = "X" * 10_000
        result = _truncate_with_head_tail(text, max_chars=500)
        assert len(result) <= 500
```

### 6.3 Parallel Scout smoke (mocked LLM)

```python
class TestScoutParallelChunks:
    def test_parallel_preserves_order(self):
        # Mock _extract_sentences_from_chunk to return [chunk_index]
        # Construct 5 chunks; run with scout_max_workers=4
        # Assert all_sentences == [1, 2, 3, 4, 5] (order preserved by executor.map)
```

### 6.4 `_is_quota_error_and_backoff`

```python
class TestQuotaErrorBackoff:
    def test_returns_false_for_non_quota_error(self):
        # ValueError → returns False, no sleep

    def test_returns_true_and_sleeps_for_429(self):
        # Patch time.sleep; assert sleep was called and returns True
```

Total: ~3–5 new test methods, ~80–100 lines.

---

## 7. Implementation Order (preview — final plan in writing-plans skill)

Each step is its own commit. Branch: `feat/EnhancedDocumentParserModule`. Commit prefix: `refactor(architect):` or `fix(architect):`.

| Step | Action | Risk |
|---|---|---|
| 1 | P3 — registry-driven temp/seed (4 sites) | Low (mechanical) |
| 2 | B1 — `_safe_response_text` helper + 4 sites | Low |
| 3 | B2 — `_check_response_completion` MAX_TOKENS accept-truncated; closes `architect-bug-001` | Low |
| 4 | B3 — `_parse_json_response` return type | Trivial |
| 5 | B4 — `analyze_document` log truth | Trivial |
| 6 | B5 — `_handle_quota_error` rename to `_is_quota_error_and_backoff` (+ 4 caller updates) | Low–Medium (4 call sites) |
| 7 | P4 — `_truncate_with_head_tail` helper + 2 sites | Low |
| 8 | P1 — `min_delay` configurable (env var + kwarg) | Low |
| 9a | P2 prerequisite — add `threading.Lock` to `TokenTracker.track_api_call` (mutation block) | Low (5-line addition, no test changes) |
| 9b | P2 — parallel Scout (rate-limit lock + ThreadPoolExecutor + opt-in default-1 worker) | Medium (real new code path) |
| 10 | New tests (B1, P4, P2 smoke, B5, TokenTracker concurrency smoke) + final regression | Low |

---

## 8. Acceptance Criteria

- [ ] All 5 bug fixes (B1–B5) applied in their named locations.
- [ ] `architect-bug-001` TODO block removed from `_check_response_completion`.
- [ ] No `temperature=0.05` literal anywhere in `architect.py` (registry-driven).
- [ ] No `seed=42` literal anywhere in `architect.py` (registry-driven).
- [ ] No `min_delay = 6.0` literal in `__init__` (replaced by env-var + kwarg resolution).
- [ ] `_truncate_with_head_tail` helper exists; both Architect and Specialist truncations use it.
- [ ] `extract_domain_sentences` runs sequentially when `scout_max_workers <= 1` (default) and in parallel via `ThreadPoolExecutor` when `scout_max_workers > 1`.
- [ ] `_wait_for_rate_limit` is thread-safe under `threading.Lock`.
- [ ] `TokenTracker.__init__` gains a `threading.Lock`; `track_api_call` mutation block is lock-protected.
- [ ] All 71 existing tests still pass.
- [ ] 4–6 new tests added (helper coverage + parallel smoke + TokenTracker concurrency smoke).
- [ ] `from core.architect import DomainArchitect; DomainArchitect()` (with `GEMINI_API_KEY` set or mocked) instantiates cleanly without a model arg, prints the registry-resolved model name in the banner.
- [ ] No `LLMConfig` references anywhere (verified by grep).
- [ ] File line count: `architect.py` may grow ≤ 50 lines net (target: stay ≤ 920). Existing 879-over-300-cap violation acknowledged and **not fixed in this scope**.

---

## 9. Risks & Mitigations

| Risk | Likelihood × Impact | Mitigation |
|---|---|---|
| Parallel Scout chunks introduce race in `last_request_time` | Medium × High (would silently violate Gemini RPM and trigger quota errors) | `threading.Lock` around `_wait_for_rate_limit`'s body; smoke test with `scout_max_workers=4` and mocked LLM verifies ordering is preserved |
| Parallel Scout chunks race `TokenTracker.stats` counter mutations | High × Medium (lost updates → underreported call counts and costs) | Step 9a: add `threading.Lock` to `TokenTracker.__init__`; wrap mutation block in `track_api_call`. Smoke test: 8 threads × 100 calls = 800 expected. |
| `MAX_TOKENS` accept-truncated lets malformed JSON through | Low × Medium (caller sees `{"error": "json_parse_failed"}` and falls back) | Existing fallback paths absorb it; additionally, console message clearly surfaces "accepting partial response" so the operator knows |
| Head+tail truncation drops middle concepts that matter | Medium × Medium (new behavior; could hide concepts vs. current head-only) | Marker text in the truncation tells the LLM the document is truncated; 60/40 ratio biased toward intro; if a real domain shows late-doc concepts being missed, ratio is one-line tunable |
| `DDD_MIN_DELAY_SECONDS=0.5` exhausts free-tier quota | Low × Low (opt-in env var; user owns the consequences) | README documents the env var with a warning |
| Test refactor lands in `test_unit.py` and pushes it over its existing cap | Medium × Low (test_unit.py already at ~470 lines — over cap pre-existing) | Put new tests in a fresh file `tests/test_architect_helpers.py` |
| Public API change accidentally breaks `main.py` or other callers | Low × High | New `__init__` kwargs are `Optional[T] = None` defaults; pre-existing `DomainArchitect()` calls unaffected. Smoke test (`from core.architect import DomainArchitect; DomainArchitect()`) in step 9. |

---

## 10. Out of Scope (explicit)

- **File split / module decomposition** (Q1, Q2, A1, A3). The 879-line god-class violation persists. Reserved for option C/D scope.
- **`_save_intermediate` opt-out** (Q4). Continues to write per-stage JSON unconditionally.
- **Logging migration** (Q3). `print()` remains throughout.
- **Magic-number elimination beyond what each fix requires** (Q6). Hardcoded retry count `5`, fallback context name `"CoreDomain"`, chunk size `10000`, max chars `50000` / `60000` left alone.
- **Provider abstraction / swappable LLM client** (A2). Reserved for WP-01a in `INDEX.md`.
- **SHA-keyed caching of pipeline runs** (P5). Reserved for option C/D scope.
- **Smarter token-aware chunking, sliding-window summarization** (P4 alternatives). The chosen head+tail strategy is intentionally simple.

---

## 11. Tracking

- **Spec:** this file (`todos/05-architect-bugfix-perf-spec.md`).
- **Companion docs:** `todos/00-context-report.md` (project context), `todos/INDEX.md` (broader EMSE plan).
- **Implementation plan:** to be produced by `superpowers:writing-plans` after this spec is approved.
