# WP-NEW-B: Schema-Conformance Probe (6-Model Smoke)

**Owner:** Baran
**Depends-on:** [WP-01a]
**Effort:** S (~2-3 days)
**Status:** TODO (NEW WP, audit-driven)
**Addresses:** [D1 6-model selection — verifies which models support strict Pydantic JSON]
**Refs:** `MASTER_PLAN.md` §3 D1, json_failed metric

---

## Goal

6 modelin (G1, G2, O1, O2, O3, O4) **Pydantic strict JSON output** uyumluluğunu sistematik olarak test et. 1-defalık aktivite. Sonuçları paper appendix'e koy.

**Neden bağımsız WP**: 
- WP-01a 9-commit sequence içinde son commit (commit 9) bunu içeriyor
- Ancak ayrıca WP olarak track ediyoruz çünkü çıktısı **paper'a girer** (json_failed metric için reference baseline)
- Reviewer'a "Hangi OSS modeli çalışır, hangisi schema kayar?" sorusunun cevabıdır

---

## Architecture

### CLI

```bash
python -m core.llm.schema_probe --models all --schemas basic,medium,complex
```

Output:
- `runs/probe_results.json` — structured results
- Console summary table

### 3 Test Schemas (varying complexity)

**Basic schema** (flat dict, 3 fields):
```python
class BasicViolation(BaseModel):
    violation_type: str  # "V1" | "V2" | ... | "V6"
    file: str
    line: int
```

**Medium schema** (nested object, 1 level):
```python
class CodeLocation(BaseModel):
    file: str
    line: int
    column: int

class MediumViolation(BaseModel):
    violation_type: str
    location: CodeLocation
    description: str
    confidence: float  # 0.0 to 1.0
```

**Complex schema** (deep nesting, lists, enum):
```python
from enum import Enum

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class CodeContext(BaseModel):
    surrounding_lines: list[str]
    imports: list[str]
    class_name: str | None

class ComplexViolation(BaseModel):
    violation_type: str
    locations: list[CodeLocation]
    severity: Severity
    description: str
    context: CodeContext
    suggested_fixes: list[str]
    confidence: float
    related_violations: list[str]  # IDs of related violations
```

### Test prompt

```
You are a DDD violation detector. Given the following code snippet, identify
any DDD violations and respond with a JSON object matching the schema.

Code:
```python
class CustomerManager:  # Should be CustomerService per DDD
    def update_user(self, user_id):  # 'user' is banned term, use 'Customer'
        ...
```

Respond ONLY with valid JSON matching the requested schema.
```

### Per-model test

For each (model, schema) pair:
1. Send prompt with `response_format={"type": "json_schema", ...}`
2. Try to parse response with Pydantic
3. Record:
   - `parsed_successfully: bool`
   - `parse_error: str | None`
   - `response_text: str` (raw)
   - `latency_ms: float`
   - `tokens_in/out: int`

Repeat 3 times per (model, schema) for variance.

### Output format

```json
{
  "probe_run_id": "probe_2026-05-09T10:00",
  "models": ["gemini-3.1-pro-preview", "gemini-3.1-flash-lite", "gpt-oss:120b-cloud", ...],
  "schemas": ["basic", "medium", "complex"],
  "results": [
    {
      "model": "gemini-3.1-pro-preview",
      "schema": "basic",
      "trial": 1,
      "parsed_successfully": true,
      "parse_error": null,
      "latency_ms": 2150.4,
      "tokens_in": 230,
      "tokens_out": 85
    },
    ...
  ],
  "summary": {
    "gemini-3.1-pro-preview": {"basic": "3/3", "medium": "3/3", "complex": "3/3"},
    "qwen3-coder-next:cloud": {"basic": "3/3", "medium": "2/3", "complex": "0/3"},
    ...
  }
}
```

### Output table for paper appendix

```
| Model                    | Basic   | Medium  | Complex |
|--------------------------|---------|---------|---------|
| gemini-3.1-pro-preview   | 3/3 ✓   | 3/3 ✓   | 3/3 ✓   |
| gemini-3.1-flash-lite    | 3/3 ✓   | 3/3 ✓   | 2/3 ⚠   |
| gpt-oss:120b-cloud       | 3/3 ✓   | 3/3 ✓   | 3/3 ✓   |
| qwen3-coder-next:cloud   | 3/3 ✓   | 2/3 ⚠   | 0/3 ✗   |
| minimax-m2:cloud         | 3/3 ✓   | 3/3 ✓   | 2/3 ⚠   |
| gemma4:31b-cloud         | 3/3 ✓   | 1/3 ⚠   | 0/3 ✗   |
```

This becomes paper §4.6 "Model Schema Compliance" subsection or §A.2 appendix.

---

## Acceptance Criteria

- [ ] `core/llm/schema_probe.py` CLI implemented
- [ ] 3 test schemas defined (basic, medium, complex Pydantic models)
- [ ] 6 models × 3 schemas × 3 trials = 54 calls
- [ ] `runs/probe_results.json` produced
- [ ] Console summary table prints
- [ ] Tests: `tests/test_llm/test_schema_probe.py` (mock OpenAI/genai responses) — at least 5 unit tests
- [ ] Integration smoke (manual, with real keys): all 6 models tested successfully on basic schema at minimum

---

## Implementation Steps

### Step 1 — Define schemas + prompt template
- 3 Pydantic schemas in `core/llm/probe_schemas.py`
- Standard prompt in `core/llm/probe_prompt.py`
- Test fixture: 1 small code snippet for prompt content

### Step 2 — Probe runner
- `ProbeRunner` class that takes model + schema, runs 3 trials, returns ProbeResult
- Uses existing `OllamaClient` and `GeminiClient` from WP-01a
- Wraps in try/except to catch JSON parse failures

### Step 3 — CLI + summary
- `argparse` for `--models`, `--schemas`, `--trials`, `--output`
- Run all combinations
- Write JSON output, print Markdown summary table

### Step 4 — Tests
- Mock the LLM clients to return known responses
- Test happy path (parse succeeds), test parse failure, test timeout, test API error

### Step 5 — Integration smoke
- Run with real keys on staging (after WP-01a integration testing)
- Adjust prompts/schemas if any model has 0/3 issues at the basic level

---

## Outputs (file paths)

- `extension/backend/core/llm/schema_probe.py`
- `extension/backend/core/llm/probe_schemas.py`
- `extension/backend/core/llm/probe_prompt.py`
- `extension/backend/tests/test_llm/test_schema_probe.py`
- `runs/probe_results.json` (when run, post-WP-01a)

**Paper integration**:
- New subsection or appendix in `paper.tex` referencing `runs/probe_results.json`
- Table generated by `scripts/render_probe_table.py` (eklenir)

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| qwen3-coder-next JSON mode unstable | Yüksek | Already known concern; document failure rate; add prompt-engineering fallback |
| gpt-oss-120b reasoning tokens leak into JSON | Orta | Use `reasoning_effort=low` config |
| Some Ollama Cloud models may not support `response_format` | Orta | If model fails on basic schema, try "instruct in prompt" fallback (json mode plain prompt) |
| Different Pydantic schema serialization across providers | Düşük | Use `model.model_json_schema()` (standard); test compatibility |

---

## Sync Points

- **WP-NEW-B blocked by WP-01a commit 5** (GeminiClient + OllamaClient ready)
- Output **directly informs paper §4** (model selection rationale + json_failed metric baseline)
