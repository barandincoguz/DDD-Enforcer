# Architect.py Bug Fixes + Performance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close 5 bugs and land 4 bounded performance wins on `extension/backend/core/architect.py` without breaking the public API or default behavior.

**Architecture:** All changes stay within `core/architect.py` and `core/token_tracker.py`. Two new module-level helpers (`_truncate_with_head_tail` in architect.py; `threading.Lock` in TokenTracker). Two new opt-in env-vars (`DDD_MIN_DELAY_SECONDS`, `DDD_SCOUT_MAX_WORKERS`). Public API preserved via `Optional[T] = None` defaults on new constructor kwargs.

**Tech Stack:** Python 3.13, pytest, `concurrent.futures.ThreadPoolExecutor`, `threading.Lock` (all stdlib). No new requirements.txt entries.

**Spec reference:** `todos/05-architect-bugfix-perf-spec.md` (approved).

**Branch:** `feat/EnhancedDocumentParserModule` (existing).

**Working directory:** `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer`. All `pytest` commands run from `extension/backend/`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `extension/backend/core/architect.py` | EDIT | All bug fixes + perf changes. Module gains 2 constants (`DEFAULT_MIN_DELAY_SECONDS`, `DEFAULT_SCOUT_MAX_WORKERS`), 1 helper (`_truncate_with_head_tail`), 1 method (`_safe_response_text`). Method renamed: `_handle_quota_error` → `_is_quota_error_and_backoff`. |
| `extension/backend/core/token_tracker.py` | EDIT | `__init__` gains `self._lock = threading.Lock()`. `track_api_call` mutation block (lines ~95–120) wrapped in `with self._lock:`. |
| `extension/backend/tests/test_architect_helpers.py` | CREATE | New unit tests for `_safe_response_text`, `_truncate_with_head_tail`, parallel-Scout smoke, `_is_quota_error_and_backoff`. |
| `extension/backend/tests/test_token_tracker_concurrency.py` | CREATE | New 1-test file: 8 threads × 100 calls smoke (asserts no lost updates under lock). |

---

## Task 1: Baseline — branch sanity + pre-task pytest

**Files:** None modified.

- [ ] **Step 1: Verify branch and HEAD**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" branch --show-current
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" log --oneline -1
```

Expected: branch `feat/EnhancedDocumentParserModule`; HEAD shows `03f447a` or later (the I2 `_validate_pricing_tiers` commit, OR the model-registry session's final fixup commits).

- [ ] **Step 2: Capture baseline pytest snapshot**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py tests/test_registry_snapshot.py 2>&1 | tail -3
```

Expected: `71 passed`.

- [ ] **Step 3: No commit (baseline only)**

---

## Task 2: P3 — registry-driven `temperature` + `seed` (4 stages)

**Files:**
- Modify: `extension/backend/core/architect.py:218–227, 318–327, 454–463, 596–605` (4 generate_content blocks)

- [ ] **Step 1: Read each affected block to confirm exact wording**

```bash
sed -n '215,230p' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
sed -n '315,330p' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
sed -n '450,465p' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
sed -n '590,605p' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
```

- [ ] **Step 2: Edit Scout (line ~216 area, inside `_extract_sentences_from_chunk`)**

Add a `sc = stage_config("Scout")` lookup at the very top of `_extract_sentences_from_chunk` (right after the `def` and docstring, before the `for retry in range(5):` loop). Replace the literal config block:

```python
# OLD
config=types.GenerateContentConfig(
    response_mime_type="application/json",
    temperature=0.05,  # Fully deterministic
    seed=42,  # Fixed seed for reproducibility
),

# NEW
config=types.GenerateContentConfig(
    response_mime_type="application/json",
    temperature=sc.temperature,
    seed=sc.seed,
),
```

- [ ] **Step 3: Edit Architect stage (`identify_contexts`, line ~316 area)**

Add `sc = stage_config("Architect")` right after the `for retry in range(5):` loop's docstring/intro (or just before the loop — once per call is fine). Replace the literal config block (same pattern as Step 2).

- [ ] **Step 4: Edit Specialist stage (`extract_all_contexts_details`, line ~452 area)**

Same pattern: `sc = stage_config("Specialist")` lookup; replace literal config.

- [ ] **Step 5: Edit Synthesizer stage (`synthesize`, line ~594 area)**

Same pattern: `sc = stage_config("Synthesizer")` lookup; replace literal config.

- [ ] **Step 6: Verify no temperature/seed literals remain**

```bash
grep -nE 'temperature=0\.|seed=42|seed=4[2-9]' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
```

Expected: zero hits.

- [ ] **Step 7: Run full regression**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py tests/test_registry_snapshot.py 2>&1 | tail -3
```

Expected: `71 passed`.

- [ ] **Step 8: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/core/architect.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "refactor(architect): read temperature + seed from stage_config registry"
```

---

## Task 3: B1 — `_safe_response_text` helper + 4 call sites

**Files:**
- Modify: `extension/backend/core/architect.py` — add helper method on `DomainArchitect`; replace 4 call sites at lines 235, 335, 471, 613.
- Create: `extension/backend/tests/test_architect_helpers.py` (new test file).

- [ ] **Step 1: Write failing test**

Create `extension/backend/tests/test_architect_helpers.py`:

```python
"""Unit tests for DomainArchitect helper methods.

Run: pytest tests/test_architect_helpers.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSafeResponseText:
    """DomainArchitect._safe_response_text — None-safe wrapper around response.text."""

    def _make_architect(self):
        from core.architect import DomainArchitect
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_for_test"}):
            with patch("core.architect.genai"):
                return DomainArchitect()

    def test_returns_text_when_present(self):
        a = self._make_architect()
        resp = Mock()
        resp.text = "hello world"
        assert a._safe_response_text(resp) == "hello world"

    def test_returns_empty_when_text_is_none(self):
        a = self._make_architect()
        resp = Mock()
        resp.text = None
        assert a._safe_response_text(resp) == ""

    def test_returns_empty_when_text_attr_missing(self):
        a = self._make_architect()
        resp = Mock(spec=[])  # mock with no attributes
        assert a._safe_response_text(resp) == ""
```

- [ ] **Step 2: Run test, expect 3 FAILs**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_architect_helpers.py::TestSafeResponseText -v
```

Expected: 3 failed, `AttributeError: 'DomainArchitect' object has no attribute '_safe_response_text'`.

- [ ] **Step 3: Add `_safe_response_text` method to `DomainArchitect`**

Open `extension/backend/core/architect.py`. Locate the `_check_response_completion` method (around line 802). Insert this new method directly above it (before line 802):

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

- [ ] **Step 4: Run test, expect 3 PASS**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_architect_helpers.py::TestSafeResponseText -v
```

Expected: 3 passed.

- [ ] **Step 5: Replace 4 call sites of `response.text`**

Locate each `result = self._parse_json_response(response.text)` line in `architect.py` (4 sites: ~235, ~335, ~471, ~613) and replace `response.text` with `self._safe_response_text(response)`:

```python
# OLD (4 sites)
result = self._parse_json_response(response.text)

# NEW (4 sites)
result = self._parse_json_response(self._safe_response_text(response))
```

Use the Edit tool with sufficient surrounding context per call site to make `old_string` unique.

- [ ] **Step 6: Verify no raw `response.text` calls remain in architect.py**

```bash
grep -nE 'response\.text' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
```

Expected: zero hits in code (only the docstring `# Ends with: ...{response_text[-50:]}` reference inside `_parse_json_response` may remain — that's a parameter name, not `response.text`).

- [ ] **Step 7: Run full regression**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py tests/test_registry_snapshot.py tests/test_architect_helpers.py 2>&1 | tail -3
```

Expected: 74 passed (71 + 3 new).

- [ ] **Step 8: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/core/architect.py extension/backend/tests/test_architect_helpers.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "fix(architect): add _safe_response_text — handle None response.text"
```

---

## Task 4: B2 — `_check_response_completion` MAX_TOKENS accept-truncated

**Files:**
- Modify: `extension/backend/core/architect.py:802–840` (`_check_response_completion` method body)

This task closes `architect-bug-001`.

- [ ] **Step 1: Read the current implementation**

```bash
sed -n '802,841p' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
```

Confirm the method's structure: docstring, no-candidates check, `finish_reason = response.candidates[0].finish_reason`, `if finish_reason == "STOP": return True`, else printout block, then retry decision.

- [ ] **Step 2: Replace the method body (everything after the docstring)**

Use the Edit tool. Replace the body — from `if not response.candidates:` through the final `return False` — with:

```python
        if not response.candidates:
            print("      ⚠️  No candidates in response")
            return False

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

The original `# TODO(architect-bug-001):` comment block is **completely removed** in this edit. The `MAX_OUTPUT_TOKENS` reference is gone.

- [ ] **Step 3: Verify `architect-bug-001` TODO is gone**

```bash
grep -n 'architect-bug-001\|MAX_OUTPUT_TOKENS\|LLMConfig' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
```

Expected: zero hits.

- [ ] **Step 4: Run full regression**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py tests/test_registry_snapshot.py tests/test_architect_helpers.py 2>&1 | tail -3
```

Expected: 74 passed.

- [ ] **Step 5: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/core/architect.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "fix(architect): MAX_TOKENS accepts partial response; closes architect-bug-001"
```

---

## Task 5: B3 — `_parse_json_response` return type honesty

**Files:**
- Modify: `extension/backend/core/architect.py:855` (signature + docstring of `_parse_json_response`)

- [ ] **Step 1: Read current signature and docstring**

```bash
sed -n '855,870p' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
```

- [ ] **Step 2: Replace signature and docstring**

```python
# OLD
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response. Simple strategy since we use response_mime_type='application/json'."""

# NEW
    def _parse_json_response(self, response_text: str) -> Dict[str, Any] | List[Any]:
        """Parse JSON from LLM response.

        Returns the parsed structure (dict or list — Gemini sometimes returns
        a top-level array). On parse failure returns the sentinel
        {"error": "json_parse_failed", "raw_response": <first 500 chars>}.
        Callers detect failure with:
            isinstance(result, dict) and result.get("error") == "json_parse_failed"
        """
```

Body unchanged.

- [ ] **Step 3: Run full regression**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py tests/test_registry_snapshot.py tests/test_architect_helpers.py 2>&1 | tail -3
```

Expected: 74 passed (signature change is type-only, no behavior change).

- [ ] **Step 4: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/core/architect.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "fix(architect): _parse_json_response signature reflects dict|list return"
```

---

## Task 6: B4 — `analyze_document` truthful log

**Files:**
- Modify: `extension/backend/core/architect.py:730`

- [ ] **Step 1: Replace the misleading line**

```python
# OLD
            print(f"  🎯 Ready for final synthesis")

# NEW
            print(f"  🎯 Returning {len(results)} context analyses; caller invokes synthesize() next")
```

- [ ] **Step 2: Run regression**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py tests/test_registry_snapshot.py tests/test_architect_helpers.py 2>&1 | tail -3
```

Expected: 74 passed.

- [ ] **Step 3: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/core/architect.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "fix(architect): truthful log — analyze_document does not call synthesize"
```

---

## Task 7: B5 — `_handle_quota_error` → `_is_quota_error_and_backoff`

**Files:**
- Modify: `extension/backend/core/architect.py:89–111` (method definition)
- Modify: `extension/backend/core/architect.py:263, 391, 514, 659` (4 caller sites)
- Modify: `extension/backend/tests/test_architect_helpers.py` (append 1 test)

- [ ] **Step 1: Append failing test**

Append to `extension/backend/tests/test_architect_helpers.py`:

```python
class TestQuotaErrorBackoff:
    """DomainArchitect._is_quota_error_and_backoff — explicit boolean-return semantics."""

    def _make_architect(self):
        from core.architect import DomainArchitect
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_for_test"}):
            with patch("core.architect.genai"):
                return DomainArchitect()

    def test_returns_false_for_non_quota_error(self):
        a = self._make_architect()
        result = a._is_quota_error_and_backoff(ValueError("not a quota error"), retry_count=0)
        assert result is False

    def test_returns_true_for_429(self):
        a = self._make_architect()
        with patch("core.architect.time.sleep") as mock_sleep:
            result = a._is_quota_error_and_backoff(
                Exception("429: Too Many Requests"),
                retry_count=0,
            )
        assert result is True
        mock_sleep.assert_called_once()  # backoff happened
```

- [ ] **Step 2: Run, expect 2 FAIL**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_architect_helpers.py::TestQuotaErrorBackoff -v
```

Expected: 2 failed, `AttributeError: 'DomainArchitect' object has no attribute '_is_quota_error_and_backoff'`.

- [ ] **Step 3: Replace `_handle_quota_error` definition (lines 89–111)**

Replace the entire method:

```python
# OLD
    def _handle_quota_error(self, error: Exception, retry_count: int) -> float:
        """Handle quota exceeded errors with exponential backoff."""
        error_str = str(error)
        is_quota_error = (
            "429" in error_str
            or "quota" in error_str.lower()
            or "ResourceExhausted" in str(type(error))
        )

        if not is_quota_error:
            return 0

        # Try to extract suggested retry delay
        retry_match = re.search(r"retry in (\d+\.?\d*)", error_str)
        if retry_match:
            wait_time = max(float(retry_match.group(1)), 10)
        else:
            # Exponential backoff: 15s, 30s, 60s, 120s
            wait_time = min(15 * (2**retry_count), 300)

        print(f"  ⚠️  QUOTA EXCEEDED - Backing off {wait_time:.1f}s...")
        time.sleep(wait_time)
        return wait_time

# NEW
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

- [ ] **Step 4: Update all 4 caller sites**

There are 4 callers at approximately lines 263, 391, 514, 659. Each looks like:

```python
# OLD pattern (4 sites)
            except Exception as e:
                if self._handle_quota_error(e, retry) == 0:
                    print(f"      [WARN] ... error: {e}")
                    if retry >= 4:
                        return <existing fallback>

# NEW pattern (4 sites)
            except Exception as e:
                if not self._is_quota_error_and_backoff(e, retry):
                    print(f"      [WARN] ... error: {e}")
                    if retry >= 4:
                        return <existing fallback>
```

The existing `print(f"      [WARN] ... error: {e}")` text and the existing fallback returns are PRESERVED — only the method name and the comparison change. Use Edit tool with surrounding context per call site to keep `old_string` unique.

The 4 call-site contexts (each unique enough for Edit tool):

1. **`_extract_sentences_from_chunk` (~line 263):** `[WARN] Chunk {chunk_num} error: {e}`
2. **`identify_contexts` (~line 391):** `[WARN] Context identification error: {e}`
3. **`extract_all_contexts_details` (~line 514):** `[WARN] Analysis error: {e}`
4. **`synthesize` (~line 659):** `[WARN] Synthesis error: {e}`

- [ ] **Step 5: Verify no `_handle_quota_error` references remain**

```bash
grep -n '_handle_quota_error' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
```

Expected: zero hits.

- [ ] **Step 6: Run tests**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_architect_helpers.py::TestQuotaErrorBackoff -v
```

Expected: 2 passed.

- [ ] **Step 7: Run full regression**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py tests/test_registry_snapshot.py tests/test_architect_helpers.py 2>&1 | tail -3
```

Expected: 76 passed (74 + 2 new).

- [ ] **Step 8: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/core/architect.py extension/backend/tests/test_architect_helpers.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "refactor(architect): rename _handle_quota_error to _is_quota_error_and_backoff"
```

---

## Task 8: P4 — `_truncate_with_head_tail` helper + 2 sites

**Files:**
- Modify: `extension/backend/core/architect.py` (add module-level helper; update 2 call sites at ~290, ~421)
- Modify: `extension/backend/tests/test_architect_helpers.py` (append 3 tests)

- [ ] **Step 1: Append failing tests**

Append to `extension/backend/tests/test_architect_helpers.py`:

```python
class TestTruncateWithHeadTail:
    """Module-level helper that preserves head + tail when truncating."""

    def test_no_truncation_when_under_budget(self):
        from core.architect import _truncate_with_head_tail
        assert _truncate_with_head_tail("short text", 100) == "short text"

    def test_keeps_head_and_tail(self):
        from core.architect import _truncate_with_head_tail
        text = "A" * 1000 + "B" * 1000
        result = _truncate_with_head_tail(text, max_chars=400, head_ratio=0.6)
        assert "A" in result[:200]   # head present
        assert "B" in result[-200:]  # tail present
        assert "[middle truncated" in result

    def test_total_length_within_budget(self):
        from core.architect import _truncate_with_head_tail
        text = "X" * 10_000
        result = _truncate_with_head_tail(text, max_chars=500)
        assert len(result) <= 500
```

- [ ] **Step 2: Run, expect 3 FAIL**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_architect_helpers.py::TestTruncateWithHeadTail -v
```

Expected: 3 failed, `ImportError: cannot import name '_truncate_with_head_tail' from 'core.architect'`.

- [ ] **Step 3: Add module-level helper to architect.py**

Insert this helper at module level — directly after the `ProgressCallback` type alias (around line 35), BEFORE the `class DomainArchitect:` line:

```python


def _truncate_with_head_tail(text: str, max_chars: int, head_ratio: float = 0.6) -> str:
    """Truncate `text` to `max_chars` by keeping the head and tail and
    dropping the middle, with an explicit marker so the LLM knows the
    document was truncated.

    Default ratio: 60% head / 40% tail. SRS documents typically place
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

- [ ] **Step 4: Run, expect 3 PASS**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_architect_helpers.py::TestTruncateWithHeadTail -v
```

Expected: 3 passed.

- [ ] **Step 5: Update Architect site (`identify_contexts`, ~line 290)**

Locate and replace:

```python
# OLD (around line 290)
        text = "\n".join(domain_sentences)
        max_chars = 50000
        if len(text) > max_chars:
            print(f"  ✂️  Truncating input: {len(text):,} → {max_chars:,} chars")
            text = text[:max_chars]

# NEW
        text = "\n".join(domain_sentences)
        max_chars = 50000
        if len(text) > max_chars:
            print(f"  ✂️  Truncating input: {len(text):,} → {max_chars:,} chars (head + tail preserved)")
            text = _truncate_with_head_tail(text, max_chars=max_chars)
```

- [ ] **Step 6: Update Specialist site (`extract_all_contexts_details`, ~line 421)**

Locate and replace:

```python
# OLD (around line 421)
        max_chars = 60000
        if len(sentences_text) > max_chars:
            print(f"  ✂️  Truncating input: {len(sentences_text):,} → {max_chars:,} chars")
            sentences_text = sentences_text[:max_chars]

# NEW
        max_chars = 60000
        if len(sentences_text) > max_chars:
            print(f"  ✂️  Truncating input: {len(sentences_text):,} → {max_chars:,} chars (head + tail preserved)")
            sentences_text = _truncate_with_head_tail(sentences_text, max_chars=max_chars)
```

- [ ] **Step 7: Verify no `text[:max_chars]` truncation patterns remain**

```bash
grep -nE 'text\[:max_chars\]|sentences_text\[:max_chars\]' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
```

Expected: zero hits.

- [ ] **Step 8: Run full regression**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py tests/test_registry_snapshot.py tests/test_architect_helpers.py 2>&1 | tail -3
```

Expected: 79 passed (76 + 3 new).

- [ ] **Step 9: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/core/architect.py extension/backend/tests/test_architect_helpers.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "perf(architect): truncate with head + tail to preserve late-document concepts"
```

---

## Task 9: P1 — `min_delay` configurable via env-var + kwarg

**Files:**
- Modify: `extension/backend/core/architect.py` — module constant + `__init__` signature + body

- [ ] **Step 1: Add module constant**

In `extension/backend/core/architect.py`, after the `INTERMEDIATE_DIR = ...` line (around line 31) and the `ProgressCallback = ...` type alias (around line 34), add:

```python
DEFAULT_MIN_DELAY_SECONDS = 6.0  # conservative default; free-tier safe.
                                 # Override via env var DDD_MIN_DELAY_SECONDS or constructor kwarg.
```

- [ ] **Step 2: Update `__init__` signature**

Replace:

```python
# OLD
    def __init__(self, model: Optional[str] = None, progress_callback: ProgressCallback = None):

# NEW
    def __init__(
        self,
        model: Optional[str] = None,
        progress_callback: ProgressCallback = None,
        min_delay: Optional[float] = None,
    ):
```

- [ ] **Step 3: Replace hardcoded `self.min_delay`**

In `__init__` body, replace:

```python
# OLD (around line 48)
        self.min_delay = 6.0

# NEW
        self.min_delay = (
            min_delay
            if min_delay is not None
            else float(os.getenv("DDD_MIN_DELAY_SECONDS", DEFAULT_MIN_DELAY_SECONDS))
        )
```

- [ ] **Step 4: Smoke-test `min_delay` default + override**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && GEMINI_API_KEY=fake python3 -c "
from unittest.mock import patch
with patch('core.architect.genai'):
    from core.architect import DomainArchitect

    a = DomainArchitect()
    print(f'default: {a.min_delay}')

    b = DomainArchitect(min_delay=2.5)
    print(f'kwarg: {b.min_delay}')

    import os
    os.environ['DDD_MIN_DELAY_SECONDS'] = '1.0'
    c = DomainArchitect()
    print(f'env: {c.min_delay}')
"
```

Expected:
```
default: 6.0
kwarg: 2.5
env: 1.0
```

- [ ] **Step 5: Run full regression**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py tests/test_registry_snapshot.py tests/test_architect_helpers.py 2>&1 | tail -3
```

Expected: 79 passed.

- [ ] **Step 6: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/core/architect.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "perf(architect): min_delay configurable via DDD_MIN_DELAY_SECONDS env var or kwarg"
```

---

## Task 10: 9a — `threading.Lock` on `TokenTracker.track_api_call`

**Files:**
- Modify: `extension/backend/core/token_tracker.py` (`__init__` adds lock; `track_api_call` mutation block wrapped)
- Create: `extension/backend/tests/test_token_tracker_concurrency.py` (new)

- [ ] **Step 1: Write failing concurrency test**

Create `extension/backend/tests/test_token_tracker_concurrency.py`:

```python
"""Concurrency smoke for TokenTracker — required before parallel Scout (Task 11).

Run: pytest tests/test_token_tracker_concurrency.py -v
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def _fake_response(prompt_tokens: int, completion_tokens: int):
    response = MagicMock()
    response.usage_metadata.prompt_token_count = prompt_tokens
    response.usage_metadata.candidates_token_count = completion_tokens
    response.usage_metadata.cached_content_token_count = 0
    return response


class TestTokenTrackerConcurrency:
    """track_api_call must be thread-safe; no lost updates under contention."""

    def setup_method(self):
        from core.token_tracker import TokenTracker
        TokenTracker.reset()

    def test_no_lost_updates_under_8_threads_x_100_calls(self):
        """8 threads × 100 calls each = 800 expected total_api_calls."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        n_threads = 8
        n_calls_per_thread = 100

        def worker():
            for _ in range(n_calls_per_thread):
                tracker.track_api_call(
                    _fake_response(prompt_tokens=10, completion_tokens=5),
                    stage="Validator",
                    operation="concurrent_smoke",
                )

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = [ex.submit(worker) for _ in range(n_threads)]
            for f in futures:
                f.result()  # raise any exception

        expected_total = n_threads * n_calls_per_thread
        assert tracker.stats.total_api_calls == expected_total, (
            f"Lost updates: expected {expected_total} calls, got "
            f"{tracker.stats.total_api_calls}"
        )
        assert tracker.stats.total_prompt_tokens == expected_total * 10
        assert tracker.stats.total_completion_tokens == expected_total * 5
        # Also validate the per-stage accumulator survived
        validator = tracker.tokens_for_stage("Validator")
        assert validator.call_count == expected_total
```

- [ ] **Step 2: Run test, expect FAIL (or flakey-PASS)**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_token_tracker_concurrency.py -v
```

Expected: FAILED — `total_api_calls` will be < 800 due to lost updates. (If it passes due to GIL luck, the test is still valuable; the lock added in Step 3 makes correctness deterministic.)

- [ ] **Step 3: Add `threading.Lock` to `TokenTracker.__init__`**

In `extension/backend/core/token_tracker.py`, add `import threading` near the other imports if not present. Then update `__init__` and the `track_api_call` mutation block:

```python
# At the top of the file, in the import block:
import threading

# In TokenTracker.__init__ (currently 2 lines):
    def __init__(self) -> None:
        self.stats = TokenUsageStats()
        self.session_start = datetime.now().isoformat()
        self._lock = threading.Lock()
```

- [ ] **Step 4: Wrap mutation block in `track_api_call`**

Locate the mutation block in `track_api_call` (currently lines ~95–120 of `token_tracker.py`). Wrap it in `with self._lock:`. The full updated section:

```python
        # Build the record outside the lock — no shared state used here.
        record = APICallRecord(
            timestamp=datetime.now().isoformat(),
            stage=stage,
            operation=operation,
            model_id=info.model_id,
            provider=info.provider,
            prompt_tokens=billable_prompt,
            completion_tokens=completion_tokens,
            total_tokens=billable_total,
            estimated_cost=round(cost, 8),
        )

        # Lock-protected mutations — counters and dict updates.
        with self._lock:
            self.stats.total_prompt_tokens += billable_prompt
            self.stats.total_completion_tokens += completion_tokens
            self.stats.total_tokens += billable_total
            self.stats.total_api_calls += 1

            accum_m = self.stats.by_model.setdefault(
                info.model_id,
                ModelTokenAccumulator(model_id=info.model_id, provider=info.provider),
            )
            accum_m.prompt_tokens += billable_prompt
            accum_m.completion_tokens += completion_tokens
            accum_m.cost_usd += cost
            accum_m.call_count += 1

            accum_s = self.stats.by_stage.setdefault(
                stage,
                StageTokenAccumulator(stage=stage, model_id=info.model_id),
            )
            accum_s.prompt_tokens += billable_prompt
            accum_s.completion_tokens += completion_tokens
            accum_s.cost_usd += cost
            accum_s.call_count += 1

            self.stats.call_history.append(record)
```

The exact `old_string` for Edit covers everything from `# Update totals.` (or whatever the first comment is) through `self.stats.call_history.append(record)`.

- [ ] **Step 5: Run concurrency test, expect PASS**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_token_tracker_concurrency.py -v
```

Expected: 1 passed.

- [ ] **Step 6: Run full regression to confirm no other tests broke**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py tests/test_registry_snapshot.py tests/test_architect_helpers.py tests/test_token_tracker_concurrency.py 2>&1 | tail -3
```

Expected: 80 passed (79 + 1 new).

- [ ] **Step 7: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/core/token_tracker.py extension/backend/tests/test_token_tracker_concurrency.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "fix(token-tracker): threading.Lock on track_api_call mutations — prerequisite for parallel Scout"
```

---

## Task 11: 9b — P2 parallel Scout chunks (opt-in via env / kwarg)

**Files:**
- Modify: `extension/backend/core/architect.py` (module constant, lock, `__init__`, `_wait_for_rate_limit`, `extract_domain_sentences`)
- Modify: `extension/backend/tests/test_architect_helpers.py` (append parallel smoke test)

- [ ] **Step 1: Add imports**

In `extension/backend/core/architect.py`, augment the import block. After `import time` add:

```python
import threading
from concurrent.futures import ThreadPoolExecutor
```

- [ ] **Step 2: Add module constant**

After `DEFAULT_MIN_DELAY_SECONDS = 6.0` (added in Task 9), add:

```python
DEFAULT_SCOUT_MAX_WORKERS = max(1, int(os.getenv("DDD_SCOUT_MAX_WORKERS", "1")))
```

- [ ] **Step 3: Update `__init__` signature**

```python
# OLD (post-Task-9)
    def __init__(
        self,
        model: Optional[str] = None,
        progress_callback: ProgressCallback = None,
        min_delay: Optional[float] = None,
    ):

# NEW
    def __init__(
        self,
        model: Optional[str] = None,
        progress_callback: ProgressCallback = None,
        min_delay: Optional[float] = None,
        scout_max_workers: Optional[int] = None,
    ):
```

- [ ] **Step 4: Add lock + scout_max_workers to `__init__` body**

After `self.min_delay = ...` (set in Task 9), add:

```python
        self._rate_limit_lock = threading.Lock()
        self.scout_max_workers = (
            scout_max_workers
            if scout_max_workers is not None
            else DEFAULT_SCOUT_MAX_WORKERS
        )
```

- [ ] **Step 5: Wrap `_wait_for_rate_limit` body in lock**

Replace the method body:

```python
# OLD
    def _wait_for_rate_limit(self):
        """Enforce minimum delay between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed
            print(f"  ⏳ Rate limiting... waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        self.request_count += 1
        print(f"  📡 API Request #{self.request_count}")

# NEW
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

- [ ] **Step 6: Update `extract_domain_sentences` chunk loop**

Locate the existing chunk-processing section (around lines 138–146):

```python
# OLD
        for i, chunk in enumerate(chunks):
            progress = int((i + 1) / len(chunks) * 100)
            print(f"  ▶️  Chunk {i + 1}/{len(chunks)} ({len(chunk):,} chars) [{progress}%]")
            self._report_progress("Scout", "in_progress", f"Processing chunk {i + 1}/{len(chunks)}", progress)
            sentences = self._extract_sentences_from_chunk(chunk, i + 1, len(chunks))
            all_sentences.extend(sentences)
```

Replace with the conditional sequential / parallel branch:

```python
        if self.scout_max_workers <= 1:
            # Sequential — preserves original behavior exactly.
            for i, chunk in enumerate(chunks):
                progress = int((i + 1) / len(chunks) * 100)
                print(f"  ▶️  Chunk {i + 1}/{len(chunks)} ({len(chunk):,} chars) [{progress}%]")
                self._report_progress(
                    "Scout", "in_progress",
                    f"Processing chunk {i + 1}/{len(chunks)}",
                    progress,
                )
                sentences = self._extract_sentences_from_chunk(chunk, i + 1, len(chunks))
                all_sentences.extend(sentences)
        else:
            # Parallel — coalesce per-chunk progress to start/end.
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

`executor.map` preserves submission order, so `all_sentences` is deterministic.

- [ ] **Step 7: Append parallel smoke test**

Append to `extension/backend/tests/test_architect_helpers.py`:

```python
class TestScoutParallel:
    """Parallel Scout chunk smoke — opt-in via scout_max_workers > 1."""

    def _make_architect(self, scout_max_workers=None):
        from core.architect import DomainArchitect
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_for_test"}):
            with patch("core.architect.genai"):
                return DomainArchitect(scout_max_workers=scout_max_workers)

    def test_default_is_sequential(self):
        a = self._make_architect()
        assert a.scout_max_workers == 1

    def test_kwarg_overrides_default(self):
        a = self._make_architect(scout_max_workers=4)
        assert a.scout_max_workers == 4

    def test_parallel_preserves_order(self):
        """ex.map preserves submission order; mocked _extract returns chunk index list."""
        a = self._make_architect(scout_max_workers=3)

        # Force chunking by feeding a long text; mock the LLM-touching method.
        def fake_extract(chunk, num, total):
            return [f"sentence_from_chunk_{num}"]

        with patch.object(a, "_extract_sentences_from_chunk", side_effect=fake_extract):
            with patch.object(a, "_save_intermediate"):
                # Long text → 3 chunks at chunk_size=10000
                long_text = "X" * 25_000
                result = a.extract_domain_sentences(long_text)

        assert result == [
            "sentence_from_chunk_1",
            "sentence_from_chunk_2",
            "sentence_from_chunk_3",
        ]
```

- [ ] **Step 8: Run new tests, expect 3 PASS**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_architect_helpers.py::TestScoutParallel -v
```

Expected: 3 passed.

- [ ] **Step 9: Run full regression**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py tests/test_registry_snapshot.py tests/test_architect_helpers.py tests/test_token_tracker_concurrency.py 2>&1 | tail -3
```

Expected: 83 passed (80 + 3 new).

- [ ] **Step 10: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/core/architect.py extension/backend/tests/test_architect_helpers.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "perf(architect): opt-in parallel Scout chunks via DDD_SCOUT_MAX_WORKERS / kwarg"
```

---

## Task 12: Final acceptance — grep checks + smoke + commit summary

**Files:** None modified.

- [ ] **Step 1: Acceptance grep — no temperature / seed / min_delay literals in architect.py**

```bash
grep -nE 'temperature=0\.|seed=42|min_delay = 6\.0' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
```

Expected: zero hits.

- [ ] **Step 2: Acceptance grep — no `LLMConfig` / `architect-bug-001` / `MAX_OUTPUT_TOKENS`**

```bash
grep -nE 'LLMConfig|architect-bug-001|MAX_OUTPUT_TOKENS' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
```

Expected: zero hits.

- [ ] **Step 3: Acceptance grep — no `_handle_quota_error`**

```bash
grep -nE '_handle_quota_error' "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
```

Expected: zero hits.

- [ ] **Step 4: Final regression**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py tests/test_registry_snapshot.py tests/test_architect_helpers.py tests/test_token_tracker_concurrency.py -v 2>&1 | tail -5
```

Expected: 83 passed.

- [ ] **Step 5: Smoke — DomainArchitect instantiates and reports the registry-resolved model**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && GEMINI_API_KEY=fake python3 -c "
from unittest.mock import patch
with patch('core.architect.genai'):
    from core.architect import DomainArchitect
    a = DomainArchitect()
    print(f'model_name = {a.model_name}')
    print(f'min_delay = {a.min_delay}')
    print(f'scout_max_workers = {a.scout_max_workers}')
"
```

Expected:
```
model_name = gemini-3.1-pro-preview
min_delay = 6.0
scout_max_workers = 1
```

- [ ] **Step 6: File line count check**

```bash
wc -l "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py"
```

Expected: ≤ 920 (architect.py was 879 at start; spec budget allows +50 net). The pre-existing AGENTS.md violation persists — out of B-scope.

- [ ] **Step 7: No commit (acceptance verification only)**

If any of Steps 1–6 fail, the task is not done. Report status and the failing step. Do not declare completion until every check passes.

---

## Self-Review

**Spec coverage:**

| Spec section | Tasks |
|---|---|
| §4.1 B1 `_safe_response_text` | Task 3 |
| §4.2 B2 MAX_TOKENS accept-truncated (closes architect-bug-001) | Task 4 |
| §4.3 B3 `_parse_json_response` return type | Task 5 |
| §4.4 B4 truthful log | Task 6 |
| §4.5 B5 `_handle_quota_error` rename | Task 7 |
| §5.1 P3 registry-driven temp/seed | Task 2 |
| §5.2 P1 configurable `min_delay` | Task 9 |
| §5.3 P2 parallel Scout (incl. thread-safety contract) | Tasks 10 + 11 |
| §5.4 P4 head + tail truncation | Task 8 |
| §6 New tests | Tasks 3 (B1×3), 7 (B5×2), 8 (P4×3), 10 (concurrency×1), 11 (parallel×3) |
| §8 Acceptance — file size, grep, regression | Task 12 |

**Type/name consistency:**

- `_safe_response_text` (Task 3) vs `self._safe_response_text(response)` (Task 3 Step 5). ✓
- `_is_quota_error_and_backoff` (Task 7 Step 3) vs all 4 caller sites (Task 7 Step 4). ✓
- `_truncate_with_head_tail` (Task 8 Step 3) — module-level — vs imports `from core.architect import _truncate_with_head_tail` (Task 8 Step 1 tests). ✓
- `scout_max_workers` (Task 11) consistent across `__init__`, attribute, env var name (`DDD_SCOUT_MAX_WORKERS`), test class. ✓
- `min_delay` (Task 9) consistent across kwarg, attribute, env var (`DDD_MIN_DELAY_SECONDS`). ✓
- `TokenTracker._lock` (Task 10) vs `with self._lock:` (Task 10). ✓

**Placeholder scan:**

- No "TBD", "TODO", "implement later", "fill in details".
- Each step has either a code block or a concrete bash command + expected output.
- Each test has the actual test code, not just "write a test".

**Out-of-scope items** (from spec §10) — none of these have a task:
- File split (Q1, Q2)
- `_save_intermediate` opt-out (Q4)
- Logging migration (Q3)
- Magic-number elimination (Q6)
- Provider abstraction (A2)
- SHA-keyed caching (P5)

This is intentional and matches the spec.

---

**Plan complete and saved to `todos/06-architect-bugfix-perf-plan.md`.** Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
