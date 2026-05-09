# WP-NEW-A: AST-Based Drift Injector (V1-V6 Quota Tool)

**Owner:** Ali
**Depends-on:** [] (independent — pure Python AST work, no LLM, no infra)
**Effort:** M (~1 week, 4-6 days)
**Status:** TODO (NEW WP, audit-driven)
**Addresses:** [D2 Yaklaşım F enabler, RQ4 enabler]
**Refs:** `MASTER_PLAN.md` §3 D2

---

## Goal

Otomatik bir Python tool yaz: **CLEAN bir Python microservice codebase**'ini input olarak al, **DDD ihlalleri inject et** (V1-V6 quota'sıyla), output olarak modified codebase + manifest döndür.

Bu tool olmadan:
- Yaklaşım F'in DRIFT-LIGHT ve DRIFT-HEAVY varyantları manuel hand-edit gerektirir (Ali için 3-6 gün × 3 domain = 9-18 gün)
- RQ4 synthetic violations da manuel injection gerektirir
- Reviewer "manual injection introduces bias" diyebilir

Tool sayesinde:
- Inject deterministic, replicable, manifest-recorded
- Replication package'a script + manifest commit edilir → reviewer aynı codebase'i 1 dakikada üretir
- LLM Guidelines G7 (suitable benchmarks) compliant

---

## Architecture

### CLI

```bash
python -m scripts.inject_drift \
    --input subjects/D1/code-clean/ \
    --output subjects/D1/code-drift-light/ \
    --quota V1=1,V2=1,V3=1,V4=1 \
    --seed 42 \
    --manifest subjects/D1/code-drift-light/_drift_manifest.json
```

Outputs:
- Modified codebase in `--output` directory (copy + edits applied)
- `_drift_manifest.json` recording every edit

### Module structure

```
scripts/
  inject_drift.py        # CLI entry point
  
core/drift/
  __init__.py
  taxonomy.py            # V1-V6 violation type definitions
  injector.py            # main DriftInjector class
  v1_synonym.py          # V1: synonym replacement
  v2_banned_term.py      # V2: banned term injection
  v3_naming.py           # V3: naming convention violation
  v4_context_boundary.py # V4: bounded context boundary leak
  v5_aggregate.py        # V5: aggregate boundary violation
  v6_domain_event.py     # V6: domain event mishandling
  manifest.py            # ManifestRecord + serialization
```

### V1-V6 Taxonomy (paper'dan)

| V | Adı | Description | Inject Strategy |
|---|-----|-------------|-----------------|
| **V1** | Synonym | Canonical olmayan terim yerine kullanılır | Bir entity'nin field/method adında "Customer" → "Client" replacement |
| **V2** | Banned Term | Domain-yasaklı terim koda sızar | "transaction" → "Payment"-canonical olduğu domain'de "transaction" injection |
| **V3** | Naming Convention | İhlal eden ad (örn. `cstmr` yerine `Customer`) | Pythonic ama domain-aware naming bozma |
| **V4** | Context Boundary | İllegal cross-context import veya cross-context method call | `from sales.order import OrderRepository` (bounded context boundary cross) |
| **V5** | Aggregate Boundary | Aggregate root dışından inner field'a doğrudan erişim | `order.line_items[0].discount_value = ...` (should go through method) |
| **V6** | Domain Event | Domain event eksik raise edilir veya yanlış kullanılır | `OrderCreated` event raise edilmesi gereken yerde silindi |

### DriftInjector (core class)

```python
@dataclass
class ViolationConfig:
    v_type: str              # "V1" | "V2" | ... | "V6"
    target_file: str         # which file to modify
    target_symbol: str       # which class/method/var to modify
    original: str            # original code snippet
    modified: str            # modified code snippet
    rationale: str           # why this is a V<n> violation

@dataclass
class DriftManifest:
    domain: str              # "D1"
    drift_level: str         # "light" | "heavy"
    seed: int
    quota: dict              # {"V1": 1, "V2": 1, ...}
    violations: list[ViolationConfig]
    timestamp: str

class DriftInjector:
    def __init__(self, taxonomy: dict, seed: int):
        ...
    
    def inject(
        self,
        codebase_path: Path,
        output_path: Path,
        quota: dict[str, int],  # {"V1": 1, "V2": 1, ...}
    ) -> DriftManifest:
        """
        1. Copy codebase from input to output
        2. For each V-type with quota > 0:
           - Find candidate files/symbols
           - Apply N edits via type-specific strategy
           - Record each edit in manifest
        3. Verify modified codebase still parses (ast.parse)
        4. Write manifest to output_path/_drift_manifest.json
        """
```

### Per-V-type strategy (high-level)

**V1 (synonym)**:
1. Read SRS glossary or extract canonical terms from class names
2. Pick a class that uses canonical term (e.g., `Customer`)
3. In one method/field, use a synonym (`Client`, `User`, `Buyer`) instead
4. Document: "Used 'Client' in method name where canonical term is 'Customer'"

**V2 (banned term)**:
1. Have a list of "banned terms" per domain (from SRS or domain knowledge)
2. Inject a banned term as a variable/comment/docstring
3. Document the location

**V3 (naming convention)**:
1. Pick a class/method/variable
2. Apply a non-PEP8 or non-domain-aware name (e.g., `cstmr`, `do_thing`, snake_case in a Class context)
3. Document

**V4 (context boundary)**:
1. Identify bounded contexts in codebase (separate package/module folders)
2. Add an import that crosses boundaries (e.g., `sales/` imports from `inventory/internals/`)
3. Document

**V5 (aggregate boundary)**:
1. Find an aggregate root (e.g., `Order` with `LineItem` children)
2. Add a method that mutates child state directly (instead of via root method)
3. Document

**V6 (domain event)**:
1. Find a state-change method (e.g., `Order.confirm()`)
2. Remove the domain event raise (e.g., comment out `self.events.append(OrderConfirmed(...))`)
3. Document

---

## Acceptance Criteria

- [ ] `scripts/inject_drift.py` CLI entry point — accepts `--input`, `--output`, `--quota`, `--seed`, `--manifest`
- [ ] `core/drift/` package with all 6 V-type modules + injector + taxonomy + manifest
- [ ] Each V-type has at least 2 inject strategies (variety, not just one pattern)
- [ ] Output codebase always parses with `ast.parse` (validation step)
- [ ] Manifest schema documented in `core/drift/manifest.py`
- [ ] Tests: `tests/test_drift/{test_v1, test_v2, ..., test_v6, test_injector, test_manifest}.py` — at least 25 unit tests
- [ ] Test fixture: `tests/fixtures/sample_clean_codebase/` (small representative Python project, ~5 files)
- [ ] CI: tests pass, linting clean
- [ ] Smoke: run on `subjects/D1/code-clean/` (when ready), produce `code-drift-light/` with quota V1=1,V2=1,V3=1,V4=1
- [ ] Manual spot-check: 5/5 random injected violations "look natural" (not obvious test artifact)

---

## Implementation Steps

### Step 1 — Taxonomy + manifest (Day 1)
- `core/drift/taxonomy.py`: V1-V6 dataclass definitions
- `core/drift/manifest.py`: `DriftManifest` + `ViolationConfig` dataclasses
- `core/drift/__init__.py`: exports
- Tests for serialization/deserialization

### Step 2 — V1 + V2 + V3 (the "lexical" violations, Day 2)
- These don't require full bounded-context analysis; they're symbol-level
- `core/drift/v1_synonym.py`: synonym dictionary + finder + replacer
- `core/drift/v2_banned_term.py`: banned term list + injector
- `core/drift/v3_naming.py`: naming pattern violator
- Tests on small fixture codebase

### Step 3 — V4 (the "structural" violation, Day 3)
- Need to identify bounded contexts (e.g., subpackages)
- Inject import that crosses boundaries
- Test: ensure injected import is syntactically valid + parses

### Step 4 — V5 + V6 (the "semantic" violations, Day 4)
- V5: aggregate boundary violation — find class with composition, inject direct child mutation
- V6: domain event — find state-change methods, remove event raise
- These need deeper AST analysis but doable with `ast.walk` + heuristics

### Step 5 — Injector orchestrator + CLI (Day 5)
- `core/drift/injector.py`: main `DriftInjector` class
- `scripts/inject_drift.py`: CLI wrapping the injector
- End-to-end test on full fixture codebase

### Step 6 — Smoke + manual validation (Day 6)
- Run on first available CLEAN codebase from WP-02b
- Manual review of 5 random injections
- Adjust strategies if needed

---

## Outputs (file paths)

- `scripts/inject_drift.py`
- `extension/backend/core/drift/__init__.py`
- `extension/backend/core/drift/taxonomy.py`
- `extension/backend/core/drift/injector.py`
- `extension/backend/core/drift/v1_synonym.py`
- `extension/backend/core/drift/v2_banned_term.py`
- `extension/backend/core/drift/v3_naming.py`
- `extension/backend/core/drift/v4_context_boundary.py`
- `extension/backend/core/drift/v5_aggregate.py`
- `extension/backend/core/drift/v6_domain_event.py`
- `extension/backend/core/drift/manifest.py`
- `extension/backend/tests/test_drift/test_v1.py`
- `extension/backend/tests/test_drift/test_v2.py`
- ... (similar for V3-V6)
- `extension/backend/tests/test_drift/test_injector.py`
- `extension/backend/tests/test_drift/test_manifest.py`
- `extension/backend/tests/fixtures/sample_clean_codebase/`

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| V5/V6 (semantic violations) hard to inject programmatically | Yüksek | Start with simple heuristics (e.g., V5 = direct child field assignment, V6 = remove `events.append` lines); refine if too rigid |
| Injected codebase doesn't parse | Düşük | `ast.parse` validation step at end; if fail, retry with different target |
| Injection patterns too "obvious" — looks like test fixture | Orta | Manual spot-check protocol; if 5/5 look fake, refine strategies |
| Quota requested > available targets in codebase | Düşük | If codebase has only 2 places suitable for V1, can't inject 5 V1; raise warning, deliver what's possible, document in manifest |
| Random seed doesn't reproduce | Düşük | Use Python `random.seed()` consistently; test reproducibility |

---

## Sync Points

- **End of WP-NEW-A (~end of W4)**: Tool ready → input for WP-02c (DRIFT-LIGHT/HEAVY generation)
- **WP-06 dependency**: RQ4 synthetic seeding also uses this tool with **separate quota config** (no overlap with drift corpus, prevents test-set leakage)
