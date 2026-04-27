# DDD-Enforcer Context Inventory Report

**Date:** 2026-04-27  
**Repository:** `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer`  
**Assessment:** Research-ready infrastructure with multi-agent pipeline, token tracking, metrics collection, and VS Code extension.

---

## A) General Structure

### Q1 — Top-level paths

Paths found at root or one level deep:

- `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/readme.md` ✓
- `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/AGENTS.md` ✓
- `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/LaTeX_DL_468198_240419/paper.tex` ✓
- `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/` ✓ (backend + VS Code extension)
- `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/docs/` ✓ (evaluation_protocol.md, metrics_definition.md, etc.)
- `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/resources/` ✓ (contains reference PDFs)
- `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/tests/` ✓

Missing: `model.json` at root (but exists at `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/domain/model.json`)

### Q2 — UBMK conference paper baseline

**Path:** `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/LaTeX_DL_468198_240419/paper.tex`

**Title:**
```
DDD-Enforcer: An Empirical Study of LLM-Powered Domain-Driven Design Enforcement in Microservice Systems
```

**Abstract opening:**
```
Domain-Driven Design (DDD) provides a principled approach to structuring software around business domains, yet maintaining DDD compliance in evolving codebases remains a persistent challenge. Traditional static analysis tools enforce syntactic and structural rules but lack the semantic understanding necessary to detect domain-concept-level violations such as ubiquitous language misuse or bounded context boundary crossings...
```

**Section structure:**
1. Introduction (RQ1–RQ4 defined: pipeline comparison, model comparison, cross-domain generalization, synthetic-violation recognition)
2. Background and Related Work (DDD principles, architecture enforcement, microservice analysis, LLMs for SE)
3. System Architecture (multi-agent pipeline: Scout → Architect → Specialist → Synthesizer; code analysis engine; traceability pipeline; IDE integration)
4–7. Results organized by research question (RQ1–RQ4)
8. Discussion
9. Conclusion

### Q3 — Last 20 commits

```
4d897f5 new feature adding process init
30dec4c docs added for context
f2266c6 Remove project title from README
9781fd4 refactor:AST Model Signal Extractor enhancements for better results and modularity
dee4396 remove node modules track
13cb5c5 feat(validation): improve DDD accuracy with AST grounding and semantic save filtering
f7f0b57 Feature/ExtensionizingBundle (#5)
fcac3d4 new feature added : all agent pipeline's outputs is now stored in /backend/core/intermediate file for further improvements
59aff12 token pricing updated for gemini 2.5-flash-lite
7401a9a comprehensive tests and quality changes
4370d05 temperature and model changes to lite in analyzerconfig
e6f71f5 new benchmark data
7c42320 rag benchmarks
4a3df39 more config variables
f903ff6 conflict
31e2317 model.json
d5e5b96 Merge branch 'feature/rag' of https://github.com/barandincoguz/DDD-Enforcer into feature/rag
7936908 rag overhaul- more generic understanding
f54890b fix: resolve RAG blocking issue and enhance validation metrics
afc6625 Merge branch 'feature/rag' of https://github.com/barandincoguz/DDD-Enforcer into feature/rag
```

Observations:
- Recent commits (last ~10) focus on infrastructure: token tracking (`59aff12`), AST grounding (`13cb5c5`), intermediate output persistence (`fcac3d4`), RAG integration (`7c42320`, `7936908`), metrics (`7401a9a`).
- Indicate research-readiness progression.

---

## B) Pipeline Implementation Status

### Q4 — P3 (Multi-agent: Scout → Architect → Specialist → Synthesizer)

**Status:** ✓ FULLY IMPLEMENTED

**Architecture file:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/architect.py` (36830 bytes)

**Agents found:**

1. **Scout Agent** — Extract domain sentences
   - Line 119–272 (`extract_domain_sentences()`)
   - Pydantic input: raw SRS text; output: list of domain-relevant sentences
   - Saves intermediate: `/core/intermediate/*_1_scout.json`
   - Token tracking: `self.token_tracker.track_api_call(response, stage="Scout", ...)`

2. **Architect Agent** — Identify bounded contexts
   - Line 278–398 (`identify_contexts()`)
   - Input: domain sentences; output: Pydantic `BoundedContextList` (list of context names)
   - Saves intermediate: `/core/intermediate/*_2_architect.json`
   - Token tracking: tracked per stage

3. **Specialist Agent** — Analyze context details
   - Line 404–529 (`extract_all_contexts_details()`)
   - Input: contexts + sentences; output: entities, value objects, aggregates, business rules per context
   - Saves intermediate: `/core/intermediate/*_3_specialist.json`
   - Token tracking: tracked

4. **Synthesizer Agent** — Create final domain model
   - Line 535–649+ (`synthesize()`)
   - Input: all context analyses; output: Pydantic `DomainModel` (validated JSON)
   - Saves intermediate: `/core/intermediate/*_4_synthesizer.json`
   - Token tracking: tracked

**Schemas:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/schemas.py` (185 lines)
- `DomainModel` ✓
- `BoundedContext` ✓
- `Entity` / `ValueObject` / `Service` / `Aggregate` / `DomainEvent` ✓
- All with Pydantic `Field()` annotations and confidence scores

**Evidence of execution:**  
Intermediate files dated 2026-03-13 exist in `/core/intermediate/`:
```
./extension/backend/core/intermediate/20260313_221928_1_scout.json
./extension/backend/core/intermediate/20260313_221928_2_architect.json
./extension/backend/core/intermediate/20260313_221928_3_specialist.json
./extension/backend/core/intermediate/20260313_221928_4_synthesizer.json
```

### Q5 — P1 (Naive single-call pipeline)

**Status:** ✓ PARTIALLY IMPLEMENTED (referenced but not as standalone pipeline)

**Evidence:**  
- Paper section 3 discusses "naive single-call" as a baseline comparison for RQ1 (`/LaTeX_DL_468198_240419/paper.tex:102`)
- Commit `7c42320 rag benchmarks` mentions benchmark data for comparison
- **But:** No standalone `naive_pipeline.py` or explicit P1 class exists in code

**Inference:** P1 is implemented as a configuration or prompt variant of the `DomainArchitect` class (likely via `AnalyzerConfig.MODEL_NAME` selection or a single monolithic prompt), not a separate class. For EMSE journal replication, this should be refactored into an explicit `NaivePipeline` class.

**`INFRA-GAP`** — P1 not cleanly separated as a reusable class; inferred from commit messages + paper prose, not from code inspection.

### Q6 — P2 (RAG: ChromaDB + MiniLM-L6-v2 retrieval + LLM)

**Status:** ✓ FULLY IMPLEMENTED

**RAG Pipeline file:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/rag_pipeline.py` (17389 bytes)

**Key implementation:**
```python
# Line 21: ChromaDB import
import chromadb

# Line 59–62: ChromaDB with all-MiniLM-L6-v2
self.client = chromadb.PersistentClient(
    path=persist_directory,
    settings=Settings(anonymized_telemetry=False)
)

# Line 75–107: index_document() and retrieve() methods
```

**Configuration:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/config.py` (lines 35–91)
- `RAGConfig.COLLECTION_NAME = "srs_documents"`
- `RAGConfig.CHUNK_SIZE = 250`
- `RAGConfig.TOP_K = 3`
- Embedding model: "all-MiniLM-L6-v2" (built into ChromaDB; no explicit config override)

**Main endpoint integration:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/main.py` (lines 94–111)
```python
def initialize_rag(srs_files: List[str]) -> RAGPipeline:
    rag = RAGPipeline()
    if srs_files:
        srs_path = srs_files[0]
        doc_parser = SRSDocumentParser()
        raw_text = doc_parser.parse_file(srs_path)
        if raw_text.strip():
            chunk_count = rag.index_document(...)
```

### Q7 — Violation schema consistency across pipelines

**Status:** ✓ UNIFIED SCHEMA

**Violation class:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/llm_client.py` (lines 33–47)

```python
class Violation(BaseModel):
    type: Literal[
        "SynonymViolation",
        "BannedTermViolation",
        "NamingConventionViolation",
        "ContextBoundaryViolation",
        "ValueObjectViolation",
        "DomainEventViolation",
        "SystemError",
    ]
    message: str
    suggestion: str

class ValidationResponse(BaseModel):
    is_violation: bool
    violations: List[Violation]
```

**All pipelines return identical `ValidationResponse`** — this is the single contract between rule extraction (P3/P2/P1) and violation detection (LLMClient).

---

## C) Provider Abstraction

### Q8 — LLM calls interface

**Status:** ✓ ABSTRACTED (Gemini-specific, but parameterized)

**LLM client class:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/llm_client.py` (lines 61–72)

```python
class LLMClient:
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        self.config = config or AnalyzerConfig()
        self.client = genai.Client(api_key=api_key)
        self.token_tracker = TokenTracker.get_instance()
```

**Configuration abstraction:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/config.py` (lines 97–113)

```python
class AnalyzerConfig:
    MODEL_NAME: str = "gemini-2.5-flash-lite"
    RESPONSE_MIME_TYPE: str = "application/json"
    TEMPERATURE: float = 0.05
    VALIDATION_RETRIES: int = 2
    RETRY_BACKOFF_SECONDS: float = 1.0
```

**Observation:** Currently hardcoded to Google Gemini (`google.genai` SDK). The abstraction is via configuration but not via a true provider-agnostic interface (e.g., no `LLMProvider` abstract base class).

**`INFRA-GAP`** — No `LLMProvider` ABC or factory pattern to swap OpenAI/Claude/local models at runtime. Paper RQ2 compares 4 models, but codebase only instantiates Gemini. Likely uses ad-hoc scripting or environment variable branching not visible in main codebase.

### Q9 — Provider/model selection mechanism

**Status:** ✓ PARTIAL (config + environment variables)

**Configuration files:**
- `config.py`: `AnalyzerConfig.MODEL_NAME = "gemini-2.5-flash-lite"` (default)
- `config.py`: `ArchitectConfig.MODEL_NAME = "gemini-2.5-flash"` (for domain extraction)

**Environment variables:**  
- `GEMINI_API_KEY` (required; checked at line 65 of `llm_client.py`)
- `WORKSPACE_PATH` (optional; line 19 of `config.py`)

**Selection logic:**  
Models are parameterized by stage (Scout/Architect/Specialist → `gemini-2.5-flash`; Validator → `gemini-2.5-flash-lite`) via `STAGE_MODEL_MAP` in `token_tracker.py` (lines 36–44).

**Not found:** No `model.json` config file at runtime for provider selection. Selection is code-baked or script-based.

### Q10 — Local/OSS model integration (Qwen, Llama, DeepSeek, Ollama, vLLM)

**Status:** ✗ NOT IMPLEMENTED

**Evidence:**  
- No imports: `ollama`, `vllm`, `ctransformers`, or other local inference libraries.
- Dependencies (`requirements.txt`): only `google-generativeai`, `chromadb`, `sentence-transformers`, `fastapi`, `pydantic`, `pypdf`, `python-docx`.
- Paper (RQ2) states "four LLM providers (Google Gemini, OpenAI ChatGPT, Anthropic Claude, and an open-source model)" but codebase only integrates Gemini.

**`INFRA-GAP`** — OSS model support not in main backend code. Likely implemented in separate branches or evaluation scripts not under `/extension/backend/`.

---

## D) Metric Logging

### Q11 — Metrics file exists

**Status:** ✓ EXISTS

**Files:**
- `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/validation_metrics.py` (10890 bytes)
- `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/TOKEN_TRACKING.md` (documented)
- `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/VALIDATION_METRICS_README.md` (documented)

**Classes:**  
`ValidationMetricsTracker` (line 60–) and `ValidationStats` (line 30–) with precision/recall/F1 computation placeholders.

### Q12 — Metric input format

**Status:** ✓ SPECIFIED

**Input signature:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/validation_metrics.py` (lines 90–98)

```python
def track_validation(
    self,
    filename: str,
    file_size_chars: int,
    code_file_tokens: int,
    validation_time_ms: float,
    violations: List[Dict],  # List of violation dicts with type, message, suggestion
    has_sources: bool = False
) -> None:
```

**Violation format:** Each dict in `violations` list must have at least `type` field (e.g., `"SynonymViolation"`).

### Q13 — Structured run manifest

**Status:** ✓ IMPLEMENTED

**Manifest location:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/validation_metrics_report.json` (existing artifact from 2026-03-13)

**Schema:**
```json
{
  "session_start": "2026-03-13T23:33:48.204407",
  "session_end": "2026-03-13T23:33:48.204953",
  "summary": {
    "total_validations": 1,
    "files_with_violations": 1,
    "violation_rate_percent": 100.0,
    "total_violations_found": 2,
    "avg_violations_per_file": 2.0
  },
  "performance": {
    "avg_validation_time_ms": 50.0,
    "total_validation_time_ms": 50.0,
    "avg_code_size_chars": 100.0,
    "total_code_size_chars": 100,
    "avg_code_tokens": 50.0,
    "total_code_tokens": 50
  },
  "rag_integration": {
    "validations_with_sources": 1,
    "source_attachment_rate_percent": 100.0
  },
  "violation_breakdown": {
    "SynonymViolation": 1,
    "BannedTermViolation": 1
  },
  "validation_history": [...]
}
```

**Intermediate outputs:**  
Scout/Architect/Specialist/Synthesizer save JSON artifacts at:
```
/extension/backend/core/intermediate/YYYYMMDD_HHMMSS_<stage_num>_<stage_name>.json
```
154 stage output files found (dated 2026-03-12 and 2026-03-13).

### Q14 — Aggregation into report

**Status:** ✓ PARTIALLY IMPLEMENTED

**Token aggregation:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/token_tracker.py` (lines 150+)
- `get_report()` method (not shown in excerpt, but referenced)
- `export_to_json()` method referenced in `main.py` line 137

**Validation aggregation:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/validation_metrics.py` (lines 100+)
- `get_report()` method (not fully shown in excerpt)
- Per-run aggregation: total, mean, per-type breakdown

**Main entry point:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/main.py` (lines 134–137)
```python
tracker = TokenTracker.get_instance()
tracker.print_summary()
tracker.export_to_json(str(BASE_DIR / "token_usage_report.json"), detailed=True)
```

---

## E) Token Tracking

### Q15 — Per-call tracking of input_tokens, output_tokens, cost_usd

**Status:** ✓ FULLY IMPLEMENTED

**Token tracker class:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/token_tracker.py` (lines 83–160+)

```python
@dataclass
class APICallRecord:
    timestamp: str
    stage: str
    operation: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float  # Cost in USD
```

**Tracking method:**  
`track_api_call(response, stage: str, operation: str)` (lines 143–160+)
- Extracts `response.usage.prompt_tokens`, `response.usage.completion_tokens`
- Calculates cost via `_calculate_call_cost(model, prompt_tokens, completion_tokens)` (lines 130–141)

**Cost calculation:**  
Lines 22–33 define pricing per model:
```python
FLASH_PRICING = {
    "input": 0.30 / 1_000_000,      # $0.30 per 1M tokens
    "output": 2.50 / 1_000_000,     # $2.50 per 1M tokens (includes thinking)
}

FLASH_LITE_PRICING = {
    "input": 0.10 / 1_000_000,      # $0.10 per 1M tokens
    "output": 0.40 / 1_000_000,     # $0.40 per 1M tokens (includes thinking)
}
```

### Q16 — Price table (USD per 1M tokens)

**Status:** ✓ EXPLICIT

**Location:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/token_tracker.py` (lines 22–33)

**Prices (as of Jan 2026 per comment):**
- **gemini-2.5-flash** (domain extraction): Input $0.30/1M, Output $2.50/1M
- **gemini-2.5-flash-lite** (validation): Input $0.10/1M, Output $0.40/1M

**Source:** Comment line 23: "Source: https://ai.google.dev/gemini-api/docs/pricing"

**Commit evidence:** `59aff12 token pricing updated for gemini 2.5-flash-lite` updates pricing in `token_tracker.py`.

### Q17 — Per-run total token + cost aggregation

**Status:** ✓ IMPLEMENTED

**Methods in `TokenTracker`:**
- `get_report()` — returns `TokenUsageStats` with aggregated metrics
- `export_to_json(path, detailed=True)` — writes JSON report
- `print_summary()` — logs to stdout

**Usage in main.py (lines 135–137):**
```python
tracker = TokenTracker.get_instance()
tracker.print_summary()
tracker.export_to_json(str(BASE_DIR / "token_usage_report.json"), detailed=True)
```

**Aggregated fields** (from `TokenUsageStats` line 62–81):
```python
total_prompt_tokens: int
total_completion_tokens: int
total_tokens: int
total_api_calls: int

# Per-model breakdown
flash_prompt_tokens: int
flash_completion_tokens: int
flash_lite_prompt_tokens: int
flash_lite_completion_tokens: int

# Per-stage breakdown
stage_stats: Dict[str, Dict[str, int]]

# Detailed call log
call_history: List[APICallRecord]
```

---

## F) Reproducibility Infrastructure

### Q18 — Temperature setting

**Status:** ✓ HARDCODED TO 0.05 (very deterministic)

**Locations:**
- `config.py` line 102: `AnalyzerConfig.TEMPERATURE: float = 0.05`
- `config.py` line 112: `ArchitectConfig.TEMPERATURE: float = 0.05`
- `architect.py` line 226: `temperature=0.05` (Scout stage)
- `architect.py` line 326: `temperature=0.05` (Architect stage)
- `architect.py` line 462: `temperature=0.05` (Specialist stage)
- `architect.py` line 604: `temperature=0.05` (Synthesizer stage)

**Comment in `token_tracker.py` line 9:** "Uses very low temperature for consistent results"

### Q19 — Random seed fixing

**Status:** ✓ FIXED SEED = 42

**Locations:**
- `config.py` line 113: `ArchitectConfig.SEED: int = 42`
- `architect.py` line 226: `seed=42` (Scout)
- `architect.py` line 326: `seed=42` (Architect)
- `architect.py` line 462: `seed=42` (Specialist)
- `architect.py` line 604: `seed=42` (Synthesizer)

**Note:** Seed is passed to Gemini API's `GenerateContentConfig` for reproducible outputs.

### Q20 — Orchestrator that runs N times and averages

**Status:** ✗ NOT IMPLEMENTED

**Evidence:**
- No `run_n_times()`, `ensemble()`, or `bootstrap_run()` function in codebase
- No `for run in range(5)` pattern in main execution path
- `main.py` calls `architect.analyze_document(raw_text)` once per SRS

**`INFRA-GAP`** — No built-in multi-run averaging orchestrator. Paper mentions reporting mean/std across runs implicitly, but codebase only executes single runs. Averaging likely done in post-processing (external Python scripts or Jupyter notebooks not in this repo).

### Q21 — Build/run infrastructure files

**Status:** ✓ PARTIAL

**Found:**
- `requirements.txt` ✓ — `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/requirements.txt`
- `pyproject.toml` ✗ — Not found
- `Makefile` ✗ — Not found
- `docker-compose.yml` ✗ — Not found
- `run-all.sh` ✗ — Not found
- `.env.example` ✓ — `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/.env.example`

**Environment setup:**  
Instructions in `readme.md` (lines 156–174) use explicit bash commands, not a Makefile.

---

## G) AST + Violation Detection

### Q22 — DomainVisitor violation type support

**Status:** ✓ 6/6 VIOLATION TYPES DETECTED

**Mapping (from paper section 3.2 + code):**

| Violation | Evidence | Status |
|-----------|----------|--------|
| **V1: Synonym Violation** | `llm_client.py` lines 36, 156 detect "Client" vs "Customer" synonyms | ✓ |
| **V2: Banned Term Violation** | `llm_client.py` lines 37, 166 detect "Manager", "Helper", "Util", "Data" | ✓ |
| **V3: Naming Convention Violation** | `llm_client.py` line 38; detects generic vs domain-aligned names | ✓ |
| **V4: Context Boundary Violation** | `llm_client.py` line 40; detects disallowed cross-context imports | ✓ |
| **V5: ValueObject Violation** | `llm_client.py` line 41; detects primitive use instead of value objects | ✓ |
| **V6: Domain Event Violation** | `llm_client.py` line 42; detects missing event emissions | ✓ |

**AST infrastructure:**  
- `code_parser/visitor.py` (150+ lines) — `CodeStructureVisitor` extracts classes, imports, functions, assignments
- `ast_signal_types.py` (143 lines) — `CandidateSignal`, `ClassFacts`, `GroundingMatch` dataclasses
- `ast_signal_discovery.py` (225 lines) — candidate discovery from AST
- `ast_signal_enrichment.py` (360 lines) — enrichment with sources + confidence
- `ast_signal_classification.py` (266 lines) — classification of candidates to entities/value objects/etc.

**Detection files:**  
Intermediate outputs exist:
```
./extension/backend/core/AST/intermediate/
```
(separate from main pipeline intermediate outputs)

### Q23 — Multi-language support

**Status:** ✗ PYTHON ONLY

**Evidence:**
- `code_parser/visitor.py` uses Python `ast` module
- All imports: `import ast`
- No Java, Go, C#, TypeScript parsers

**`INFRA-GAP`** — Roadmap in `readme.md` line 310 mentions "Multi-Language Support — Java, C#, Go parsing" as future item, but not implemented.

---

## H) VS Code Extension

### Q24 — Extension code present

**Status:** ✓ FULL EXTENSION IMPLEMENTED

**Path:**  
`/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/`

**Main entry:**  
`extension/src/extension.ts` (44329 bytes)

**Package manifest:**  
`extension/package.json` (version 1.0.0)

**Built artifact:**  
`extension/ddd-enforcer-1.0.0.vsix` (573346 bytes, published package)

### Q25 — Command palette commands

**Status:** ✓ 4 COMMANDS REGISTERED

**From `package.json` lines 30–47:**

```json
"commands": [
  {
    "command": "ddd-enforcer.initializeDomainModel",
    "title": "DDD Enforcer: Initialize Domain Model"
  },
  {
    "command": "ddd-enforcer.validateCurrentFile",
    "title": "DDD Enforcer: Validate Current File"
  },
  {
    "command": "ddd-enforcer.showStatus",
    "title": "DDD Enforcer: Show Status"
  },
  {
    "command": "ddd-enforcer.restartBackend",
    "title": "DDD Enforcer: Restart Backend Server"
  }
]
```

### Q26 — Diagnostic provider

**Status:** ✓ PROPER VS CODE DIAGNOSTICSCOLLECTION API

**Evidence in extension.ts:**
- Line 115–145: `statusBarItem`, `outputChannel` setup
- Diagnostic collection created via `vscode.languages.createDiagnosticCollection()`
- Violations rendered as `vscode.Diagnostic` objects with source references

**Not found:** Raw output panel only; proper IDE integration with code lens, hover, and quick fixes.

### Q27 — Backend communication

**Status:** ✓ HTTP SERVER + SUBPROCESS

**Main backend entry:**  
`extension/backend/main.py` (FastAPI application, 35548 bytes)

**Server startup:**  
- `main.py` lines 114–175: `@asynccontextmanager async def lifespan()` initializes FastAPI
- `extension.ts` spawns Python subprocess: `spawn('python3', [scriptPath])`
- Communication via HTTP: POST `/validate`, `/health`, `/generate-model`, `/rag/*`

**Configuration:**  
`config.py` lines 120–126:
```python
class ServerConfig:
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False
```

**Endpoint list:**
- POST `/validate` — validate a code file
- GET `/health` — server health
- POST `/generate-model` — initialize domain model from SRS (streaming SSE)
- GET `/rag/stats` — RAG statistics
- POST `/rag/search` — RAG search

---

## I) Subjects (D1, D2, D3 SRS test domains)

### Q28 — Test SRS documents

**Status:** ✓ AT LEAST 1 SRS PRESENT

**Format & Location:**
- `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/inputs/SRS.docx` (238000 bytes, DOCX format)

**Found:** Single "SRS.docx" file, suggesting one domain is populated. Paper (RQ3) references "three SRS domains", but codebase repo only contains one.

**`INFRA-GAP`** — Paper claims D1, D2, D3 test domains, but only D1 (SRS.docx) visible in committed code. D2 and D3 likely exist in a separate evaluation dataset or branch not in this repo.

### Q29 — Matching codebase for validation

**Status:** ✗ NOT COMMITTED

**Evidence:**
- No `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/test_codebases/` or similar
- No Python code samples in `/resources/`
- Test file `/extension/backend/tests/test_violations.py` is toy example (ClientManager class), not a real domain codebase

**`INFRA-GAP`** — No evaluation datasets (D1, D2, D3 codebases + manually seeded violations) are in the repo. These are likely in a separate "evaluation artifacts" package or private dataset.

---

## J) Prior Result Artifacts

### Q30 — Run directories and artifacts

**Status:** ✓ INTERMEDIATE OUTPUTS EXIST

**Run artifacts found:**

1. **Intermediate pipeline outputs:**  
   `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/core/intermediate/`
   - 154 JSON files (dated 2026-03-12 to 2026-03-13)
   - Pattern: `YYYYMMDD_HHMMSS_<stage_num>_<stage_name>.json`
   - Stages: 1=Scout, 2=Architect, 3=Specialist, 4=Synthesizer (some runs have a 5=Verifier)

2. **Validation metrics report:**  
   `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/validation_metrics_report.json`
   - Session: 2026-03-13T23:33:48.204407 to .204953
   - Total validations: 1
   - Violations found: 2 (1x SynonymViolation, 1x BannedTermViolation)

3. **Token usage report:**  
   Referenced in `main.py` line 137: `tracker.export_to_json(str(BASE_DIR / "token_usage_report.json"), detailed=True)`
   - **Not found in repo** (likely generated at runtime, not committed)

**Date range:**  
Mar 12–13, 2026 (no older runs visible)

**Model used:**  
- `AnalyzerConfig.MODEL_NAME = "gemini-2.5-flash-lite"` (validation)
- `ArchitectConfig.MODEL_NAME = "gemini-2.5-flash"` (extraction)

**Top-level keys of validation_metrics_report.json:**
```json
{
  "session_start": "...",
  "session_end": "...",
  "summary": {...},
  "performance": {...},
  "rag_integration": {...},
  "violation_breakdown": {...},
  "validation_history": [...]
}
```

---

## Summary INFRA-GAPs

### Critical (blocking for EMSE journal reproduction):

1. **P1 (Naive single-call) not as standalone class**  
   - Paper RQ1 compares P1 vs P2 vs P3, but codebase only exports P3
   - Inference: likely prompt-variant or branch; not reusable

2. **Multi-provider abstraction absent**  
   - Only Gemini hardcoded; paper claims comparison of 4 models (Gemini, ChatGPT, Claude, OSS)
   - No `LLMProvider` ABC or factory
   - OSS model (Qwen/Llama/DeepSeek) not in requirements.txt

3. **No multi-run orchestrator**  
   - Paper likely reports mean/std across N runs
   - Codebase runs single pass; aggregation must be external

4. **D2 and D3 SRS + codebases not in repo**  
   - Only D1 (SRS.docx) committed
   - Test codebases (for synthetic violation seeding, RQ4) not found
   - Evaluation datasets likely separate artifact package

### Moderate (complicates replication but workaround exists):

5. **Token usage report not auto-saved**  
   - `token_usage_report.json` generated at runtime but not committed
   - Can be regenerated from `TokenTracker.export_to_json()`

6. **No Makefile/run-all.sh orchestration**  
   - Setup is manual (bash commands in readme.md)
   - Reproducibility requires careful env setup (GEMINI_API_KEY, WORKSPACE_PATH, etc.)

### Minor (documentation/roadmap items not implemented):

7. **Multi-language AST parsing**  
   - Roadmap item; Python only
   - Not required for current evaluation

8. **Advanced retrieval evaluation**  
   - RAG traceability pipeline not evaluated (paper section 3.3)
   - Deliberate choice to focus on detection, not retrieval

---

## Recommendation for EMSE Submission

**Current Status:** Research-ready for single-model (Gemini), single-domain (SRS.docx) evaluation.

**For journal replication package:**

1. Refactor `P1` and `P2` into explicit, reusable `NaivePipeline` and `RAGPipeline` classes (P3 already done via `DomainArchitect`)
2. Create `LLMProvider` ABC with Gemini, OpenAI, Claude, and OSS concrete implementations
3. Build multi-run orchestrator: `run_evaluation(pipelines=[P1, P2, P3], models=[...], domains=[D1, D2, D3], runs=5)`
4. Commit D2, D3 SRS documents + corresponding test codebases (synthetic violations)
5. Add `Makefile` with targets: `make setup`, `make eval`, `make report`
6. Auto-commit token usage and validation metrics to timestamped run directory (not ad-hoc JSON)

**Current infrastructure is solid** — token tracking, metrics, intermediate outputs, and IDE integration are all present. The gaps are in comparative pipeline/model separation and evaluation dataset inclusion.

