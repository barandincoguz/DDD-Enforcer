# DDD-Enforcer → Empirical Software Engineering — Master Plan (Locked)

> **Bu dosya, projenin canonical roadmap'idir. Tüm WP-XX dosyaları, INDEX.md ve diğer planning artifact'ları buradaki kararlara tabidir. Çelişki olursa bu dosya kazanır.**
>
> **Last revision**: 2026-05-08 v2 (D1-D7 locked + audit-driven restructure)

---

## 1. Hedef

**Springer Empirical Software Engineering** dergisi (regular track, single-blind hakemlik) — `https://link.springer.com/journal/10664`. UBMK 2025 konferans makalesinin genişletilmiş dergi versiyonu. Hedef gönderim penceresi: **Ağustos-Eylül 2026**, 14 hafta.

### EMSE Bar (web research'ten doğrulandı)
- Submission ücreti yok, decision <3 ay
- **Open Science Initiative** (badge için ek review): minor-revision aşamasında replication package istenir
- LLM çalışmaları için **G1-G8 rehberi** (arXiv 2508.15503): model versiyonu, prompt logs, interaction history, human validation, açık-LLM baseline, suitable benchmarks, limitations
- Statistical rigor + 95% CI + threats-to-validity + replication package zorunlu

---

## 2. Yazar + Rater Yapısı

### Yazarlar (3)
| Yazar | Rolü |
|-------|------|
| **Baran Dincoguz** | Code + reviewer + paper-writing |
| **Ali Kendir** | Code + reviewer + paper-writing |
| **Prof. Dr. Murat Karakaya** | Supervisor, paper review, yazar olarak ekleniyor |

### Rater Ekibi (Fleiss's κ için, 3 rater)
| Rater | İlişki |
|-------|--------|
| Baran | Yazar + rater |
| Ali | Yazar + rater |
| **TEDU Bağımsız Hoca** (henüz adı belirsiz) | **External rater** — projeden bihaber |

**Murat Hoca rater DEĞİL** — supervisor + yazar olarak kalır. ESE reviewer'ları için temiz separation (supervisor double-as-rater sorunundan kaçınır).

---

## 3. Locked Decisions (D1-D7)

### D1 — Modeller (6 toplam)

**Closed (Gemini family)**:
- **G1**: `gemini-3.1-pro-preview` — frontier, Google free credit kapsamında
- **G2**: `gemini-3.1-flash-lite` — budget tier, $0.25/M input

**OSS via Ollama Cloud (OpenAI-compatible API)**:
- **O1**: `gpt-oss:120b-cloud` — OpenAI open weights, 120B reasoning-generalist, native JSON mode
- **O2**: `qwen3-coder-next:cloud` — 80B/3B-active MoE, code-specialist
- **O3**: `minimax-m2:cloud` — 230B/10B-active MoE, SWE-bench %69.4 (#1 OSS global)
- **O4**: `gemma4:31b-cloud` — Google open-weights, "same-lab open-vs-closed" karşılaştırması için ilginç

**Provider abstraction**: 2 client yeterli (`GeminiClient` + `OllamaClient`). 6 free-tier Ollama API key + 1 Gemini key. **Key rotation** + retryable error handling (429/403/5xx) zorunlu.

**Yeni metrik**: `json_failed` rate — Pydantic strict istendi ama model geçersiz JSON üretti mi? Tablo 7 (RQ2) yeni kolon.

### D2 — Domain + Codebase (Yaklaşım F)

**3 sektör**:
- **D1 — E-ticaret/marketplace**: Mevcut authored SRS (`extension/backend/inputs/SRS.docx`), repo'ya publish + license belirle
- **D2 — Bankacılık**: Externally sourced public SRS — GitHub topic `software-requirements-specification` üzerinden aday taranır, license check
- **D3 — Sağlık (hospital management / EHR)**: Aynı protokolle public SRS

**Codebase strategy — Yaklaşım F (3 varyant per domain)**:
- **CLEAN**: LLM-generated baseline (Gemini Pro ile), edit yok
- **DRIFT-LIGHT**: 3-5 ihlal injected (otomatik AST tool ile)
- **DRIFT-HEAVY**: 10-15 ihlal injected (V1-V6 dengelenmiş quota)

**Plus**: RQ4 için **ayrı seeded codebase** — drift varyantlarıyla **örtüşmez** (test-set leakage önler). 6 ihlal × 5 seed × 3 domain = 90 kontrollü ihlal.

**WP-NEW-A**: Otomatik AST drift injector (`scripts/inject_drift.py`). V1-V6 quota sistemi.

### D3 — Validator (3-Rater Fleiss's κ)

**3 rater**: Baran + Ali + TEDU bağımsız Hoca.

**Stratified sample**:
- ~150 verdict
- Quota: 25 verdict / V-type × 6 = 150
- Pipeline (P1/P2/P3) dengeli, domain (D1/D2/D3) dengeli, model (~%50 G1, ~%50 OSS-mix)
- Judge confidence: %50 high-conf + %30 disagreement + %20 borderline

**Workflow**:
1. **Calibration session**: 1 saat, 3 rater 5 örnek case üzerinde tartışır, rubric'in muğlak yerlerini netler. Pre-audit Fleiss's κ ≥ 0.70 hedef
2. **Asıl audit**: ~150 verdict, ~12-15 saat external Hoca, 2-3 hafta yayılabilir
3. **Hesaplama**: Fleiss's κ (3-rater agreement matrix) + pairwise Cohen's κ (sub-analysis)

**Hedef threshold**: Fleiss's κ ≥ 0.6 (Landis-Koch "substantial agreement"). Altında threats-to-validity'de disclose, üstünde rahatlıkla ESE rigor barını geçeriz.

**Recruitment**: Baran TEDU bağımsız Hoca'ya direkt yaklaşır (zaten tanıyor). Murat Hoca courtesy bilgilendirilir (`todos/HOCA_GUNDEM.md` Konu 1).

### D4 — N (Run Sayısı)

**N=10 baseline** + **pilot escalation**:
- **Pilot (Hafta 4)**: D1 + P3 + 6 model × N=3 quick run, F1 std ölçülür
- F1 std < 0.05 → N=5'e in (variance düşük, daha az koşum yeter)
- F1 std 0.05-0.15 → N=10 (varsayılan)
- F1 std > 0.15 → N=20 + determinism debug

**Maliyet hesabı (N=10 baseline)**:
- ~720 toplam koşum (6 model × N=10 × 3 domain × 4 RQ approximation)
- Gemini Pro koşumları: ~$45
- Flash-Lite: ~$2
- OSS via Ollama Cloud: $0
- Judge LLM: ~$50
- **Toplam: ~$100** (Google free credit $300 kapsamında bol marj)

### D5 — Refactor Close-Out (DONE)

P0-D'de doğrulandı: `core/parser.py` ve `core/ast_model_signals.py` saf facade/shim, hiç TODO yok, çağıranlar public API kullanıyor, testler yeşil. Big-bang refactor için (`core/llm/`) ayrı bir adım.

### D6 — RQ5 Silindi

**Tamamen silindi**. Reviewer'a bahsedilmez. Murat Hoca courtesy bilgilendirilir (`todos/HOCA_GUNDEM.md` Konu 2). Cover letter'da pre-emption yok. Paper sadece RQ1-RQ4 ile temiz çıkar. Gerekçe: 2 yazar + 14 hafta için 5 RQ agresif; depth over breadth tercih.

**Düşürülen WP'ler**: ~~WP-09~~ (Practitioner Survey, RQ5 ile bağlantılıydı), ~~WP-18~~ (RQ5 design + execution).

### D7 — Security Incident (False Alarm)

`.env` git geçmişinde **hiç olmamış**. `.gitignore` 4 farklı pattern ile zaten kapsıyor. Audit raporu yanılmış. Action item yok.

---

## 4. RQ Yapısı

### RQ1 — Pipeline Karşılaştırması
**Soru**: "Naïve, retrieval-augmented ve multi-agent pipeline'lardan hangisi DDD ihlal tespitinde en iyi denge sunar?"

- **3 pipeline**: P1 (single-call) vs P2 (RAG) vs P3 (multi-agent: Scout→Architect→Specialist→Synthesizer)
- **Test koşulu**: D1 üzerinde, tek model (G1 Gemini 3.1 Pro Preview)
- **Yaklaşım F sonucu**: Precision (CLEAN'de) **+** Recall (DRIFT'te)
- **Kazanan pipeline → RQ2'ye taşınır**

### RQ2 — Model Karşılaştırması (3 alt-RQ)

**RQ2a — Within-Family Scaling**: Aynı Gemini ailesinde Pro Preview vs Flash-Lite. "Pratik dağıtımda hangi tier yeterli?"
- G1 vs G2

**RQ2b — Closed vs OSS Frontier**: En iyi closed (G1) vs en iyi OSS.
- G1 vs RQ2c'den çıkan en iyi OSS

**RQ2c — OSS Landscape**: Açık-ekosistemde 4 model:
- O1 (gpt-oss reasoning-generalist) vs O2 (qwen3-coder code-specialist) vs O3 (minimax-m2 frontier-MoE) vs O4 (gemma4 compact-Google)

**Yeni kolon**: Tüm RQ2 alt-tablolarına `json_failed_rate` eklenir.

### RQ3 — Cross-Domain Generalization
"Kazanan konfigürasyon (RQ1+RQ2'den) 3 farklı sektörde benzer performans veriyor mu?"

- **3 domain × 3 codebase variant** = 9 koşum noktası
- "Feasibility across domains" iddiası (full generalization değil — proof-of-concept çerçevesi)

### RQ4 — Synthetic Violation Recognition
"Kasıtlı olarak ekilmiş ihlalleri framework geri çağırabiliyor mu?"

- 6 V-type × 5 seed × 3 domain = 90 kontrollü ihlal
- WP-NEW-A AST drift injector ile otomatik üretilir
- DRIFT-LIGHT/HEAVY corpus'larıyla **örtüşmez** (test-set leakage önlenir)
- Metrik: seeded-recall (tespit edilen / ekilmiş)

---

## 5. 6-Phase Roadmap (23 Active WP)

> **Kritik yol**: P0 (DONE) → WP-01a [chokepoint] → WP-01b → WP-02 → WP-NEW-A → RQ chain → WP-08 → WP-13 → WP-12 → WP-15

### Phase 0 — Hijyen (DONE ✅)

4 atomic commit (main):
- `4a893c8` Python 3.12 pin + requirements.txt truncation fix
- `2609001` requirements.lock (2519 satır, hash-pinned)
- `696188d` google-generativeai → google-genai SDK migration
- `56919da` CI tests + coverage gate %60 (gerçek %67) + pyright + integration marker

### Phase 1 — Çekirdek Altyapı (W2-4)

| WP | Owner | Açıklama | Effort |
|----|-------|----------|--------|
| **WP-00** | Joint | Scope lock (N=10, M=6, K=3) → `configs/scope.yaml` | S |
| **WP-01a** | Baran | Provider abstraction (`core/llm/` paketi, 9-commit TDD big-bang refactor) | M |
| **WP-01b** | Baran | Run-spec orchestrator (idempotent worker + YAML manifest, resume capability) | M |
| **WP-01c** | Baran | Token tracking + json_failed metric integration (6 model normalization) | S |
| **WP-01d** | Baran | P1/P2/P3 pipeline classes (extraction from architect.py) | M |
| **WP-NEW-B** | Baran | Schema-Conformance Probe (6 model × 3 schema smoke; 1-time activity) | S |

### Phase 2 — Veri + Eval Altyapısı (W3-5, paralel)

| WP | Owner | Açıklama | Effort |
|----|-------|----------|--------|
| **WP-02** | Ali | Subject corpus (3 domain × 3 codebase variant + sourcing). Alt-task'lar: 02a sourcing, 02b clean, 02c drift variants | L |
| **WP-NEW-A** | Ali | AST Drift Injector (`scripts/inject_drift.py`, V1-V6 quota tool, pure Python) | M |
| **WP-07** | Baran | Judge LLM rubric + cross-family selection (test edilen Gemini ise Judge minimax) | S |

### Phase 3 — RQ Yürütme (W5-9)

| WP | Owner | Açıklama | Effort |
|----|-------|----------|--------|
| **WP-03** | Baran | RQ1 (P1/P2/P3 precision-on-clean + recall-on-drift) | S |
| **WP-04** | Baran | RQ2 split: 2a + 2b + 2c, 6 model, json_failed kolon | M |
| **WP-05** | Baran | RQ3 cross-domain × 3 drift level | S |
| **WP-06** | Baran | RQ4 synthetic seeded (drift injector kullanır) | S |

### Phase 4 — Validasyon + İstatistik (W8-11, paralel)

| WP | Owner | Açıklama | Effort |
|----|-------|----------|--------|
| **WP-08** | Joint (3 rater) | 3-Judge Audit + Fleiss's κ + calibration session | M |
| **WP-17** | Ali | Pre-registered SAP + power analysis + pilot gate (Hafta 4) | M |
| **WP-NEW-C** | Ali | Prompt Sensitivity Ablation (3 prompt variant × pipeline) | S |

### Phase 5 — Yazım (W10-13)

| WP | Owner | Açıklama |
|----|-------|----------|
| **WP-10** | Ali | Bibliography (verify + 8+ yeni ref) |
| **WP-11** | Ali | Figures (4 vector PDF, RQ2 6-model Pareto) |
| **WP-13** | Baran | Discussion + threats prose (RQ5 referans temizliği, RQ2a/b/c sub-narratives, json_failed analizi) |
| **WP-14** | Ali | Abstract + conclusion polish (numerically grounded) |
| **WP-16** | Ali | Extension architecture documentation (1.5 sayfa + 3-5 screenshot) |

### Phase 6 — Replication + Submission (W13-15)

| WP | Owner | Açıklama |
|----|-------|----------|
| **WP-12** | Baran | Replication package, Zenodo DOI, GitHub release |
| **WP-NEW-D** | Ali | Reviewer/Evaluator Guide (`EVALUATION.md`, 30-dk artifact-eval recipe) |
| **WP-15** | Joint | Submission package + cover letter + EM portal |

### Phase 7 — Tampon (W15-16)

1-2 hafta. Hoca son review, replication broken-link fix, format ayar.

### Aktif scope DIŞI

- ❌ **WP-09** Practitioner Survey (RQ5 ile bağlantılı, düştü)
- ❌ **WP-18** RQ5 Design + Execution (D6'da silindi)

---

## 6. Critical Path & Bağımlılıklar

```
P0 (DONE)
  └─ WP-00 (kickoff)
      └─ WP-01a [chokepoint, 9-commit] ───┬──> WP-01b ──> WP-01c ──> WP-01d ──> WP-NEW-B
                                          │
                                          └──> WP-07 (Judge LLM)
                                          
  WP-02 (paralel başlar) ──> WP-NEW-A ──┐
                                        ├──> WP-03 → WP-04 → WP-05/06
                                        │
  WP-NEW-A ─────────────────────────────┘
  
  WP-03..06 ──> WP-08 ⊥ WP-17 ⊥ WP-NEW-C
                  ↓
              WP-13 → WP-14
                  ↓
              WP-10 ⊥ WP-11 ⊥ WP-16
                  ↓
              WP-12 → WP-NEW-D → WP-15
```

### 4 Critical Blocker (Audit'ten)

1. **WP-NEW-A (AST Drift Injector)** — RQ4 ve Yaklaşım F çalışamaz; Faz 2 chokepoint
2. **WP-08 3-Judge Setup (D3 implementation)** — TEDU external Hoca recruitment Hafta 1'de başlamazsa Hafta 8 audit kayar
3. **WP-17 N-Pilot Gate (Hafta 4)** — D1+P3+6 model pilot olmadan RQ2 batch'leri yanlış N ile koşar
4. **RQ5 Silinme Cleanup** — INDEX.md güncel artık; eski WP-XX dosyalarındaki RQ5 referansları kontrol edilir

---

## 7. WP-01a Implementation Plan (Sıradaki — TDD)

### Modül yapısı (`core/llm/`)
```
core/llm/
  __init__.py           # public API: get_client, LLMClient, errors
  base.py               # LLMResponse, LLMClient ABC, TokenUsage
  errors.py             # RateLimitError, AuthError, SchemaError, RetryExhausted
  retry.py              # @with_retry_and_rotation (saf decorator)
  registry.py           # model_id → ModelSpec (provider, name, pricing, capabilities)
  gemini.py             # GeminiClient (google-genai SDK)
  ollama.py             # OllamaClient (OpenAI SDK → ollama.com/v1)
  schema_probe.py       # CLI: 6 model × 3 schema smoke
```

### Eski kod (silinecek)
- `core/llm_client.py` (258 satır) — big-bang silme

### LLMResponse dataclass
```python
@dataclass
class LLMResponse:
    content: str
    parsed: BaseModel | None  # None = json_failed
    usage: TokenUsage
    model_id: str
    json_failed: bool
    json_fail_reason: str | None  # "schema_mismatch" | "invalid_json" | "missing_field"
    latency_ms: float
    raw_response: dict
```

### Retry + Key Rotation Pattern
```python
RETRYABLE_STATUS = {429, 403, 500, 502, 503, 504}

def with_retry_and_rotation(*, max_retries=3, base_delay=1.0):
    """
    1. attempt 0: keys[0]
    2. 429/403 → keys[1], no delay
    3. all keys 429 → exponential backoff (1s, 2s, 4s)
    4. 5xx → same key, exponential backoff
    5. 400/401 → RAISE (no retry)
    6. max_retries → RetryExhausted
    """
```

### 9-Commit TDD Sequence

| # | Commit | Test sayısı | Risk |
|---|--------|-------------|------|
| 1 | `core/llm/base.py` + `errors.py` (interface only) | ~8 | Düşük |
| 2 | `core/llm/registry.py` (6 model + pricing migration) | ~10 | Düşük |
| 3 | `core/llm/retry.py` (saf decorator, mock-test) | ~12 | Düşük |
| 4 | `core/llm/ollama.py` (yeni client + retry+rotation) | ~10 | Orta |
| 5 | `core/llm/gemini.py` (yeni client + retry) | ~10 | Orta |
| 6 | `core/architect.py` migration (eski → yeni gemini.py) | mevcut testler güncellenir | **YÜKSEK** |
| 7 | `main.py` + diğer caller migration | regression | Yüksek |
| 8 | `core/llm_client.py` SİL | tüm test geçer | Düşük |
| 9 | `core/llm/schema_probe.py` + 6-model smoke | yeni test | Düşük |

Her commit yeşil-CI + ara state rollback'e izin verir.

---

## 8. Verification Checklist (Submission-Ready Tanımı)

### Faz 0 (DONE) ✅
- [x] `gitleaks detect` 0 finding
- [x] `pip install -r requirements.lock && pytest -m "not integration"` temiz repo'da geçer
- [x] CI yeşil + coverage %67 (gate %60)

### Faz 1 sonu
- [ ] `core/llm/` paketi test-driven complete
- [ ] 6-key Ollama rotation + retry decorator çalışır
- [ ] `core/llm_client.py` silinmiş, regression yok
- [ ] `pyright` 0 error

### Faz 4 sonu
- [ ] Fleiss's κ ≥ 0.6 tüm V1-V6'da (ya da threats'ta açıkça disclose)
- [ ] Bootstrap 95% CI tüm precision/recall/F1 hücrelerinde
- [ ] Wilcoxon p-values + Holm-corrected RQ1'de
- [ ] Friedman p + Nemenyi posthoc RQ2'de
- [ ] Cliff's δ tüm pairwise comparisons'da
- [ ] N pilot gate sonucu locked

### Submission
- [ ] `grep -rn PLACEHOLDER paper.tex` 0 sonuç
- [ ] `grep -rn "RQ5" paper.tex` 0 sonuç (D6 silme doğrulaması)
- [ ] Tüm RQ tabloları auto-render (manuel data entry yok)
- [ ] `json_failed_rate` kolonu Tablo 7'de
- [ ] Zenodo DOI mint edilmiş
- [ ] EVALUATION.md ile evaluator 30 dk içinde Tablo 5 reprod edebilir
- [ ] Cover letter draft (RQ5 hiç bahsetmez)
- [ ] Suggested reviewers (3-5 isim, conflict-free)
- [ ] Murat Hoca final go (5-day async review)

---

## 9. Open Questions / Açık Konular

| # | Konu | Tartışılacak yer | Sahip |
|---|------|------------------|-------|
| Q1 | Gemma versiyonu — gemma3 vs gemma4? | DONE: gemma4:31b-cloud locked (D1) | — |
| Q2 | Hocayla RQ1 pipeline count (2 vs 3) | `HOCA_GUNDEM.md` Konu 3 | Baran |
| Q3 | TEDU external Hoca'nın resmi adı | `HOCA_GUNDEM.md` Konu 1 | Baran |
| Q4 | RQ5 silmeyi Hocaya bildirme | `HOCA_GUNDEM.md` Konu 2 | Baran |
| Q5 | D2 (banking) ve D3 (healthcare) için seçilen public SRS adayları | WP-02 phase 02a | Ali |
| Q6 | Pilot N decision (Hafta 4) | WP-17 SAP doc | Ali (impl) + Baran (decision) |

---

## 10. Değişiklik Geçmişi

- **2026-04-27** v0: Konferans paper'ından dergi versiyonuna geçiş kararı, 18 WP envanter
- **2026-05-07** v1: İlk plan + threat register + 18 WP + 4 NEW (D6 = "future work"; D7 P0 verisi)
- **2026-05-08** v2: Audit sonrası restructure. D1-D7 locked. RQ5 silindi. Yaklaşım F (3 codebase variant). 3-rater Fleiss's κ. 23 active WP. Author + rater team netleşti. WP-01a TDD ready

---

## Cross-References

| Konu | Dosya |
|------|-------|
| Hızlı entry point | `todos/AGENT_QUICKSTART.md` |
| Status board | `todos/INDEX.md` |
| WP allocation | `todos/WP_DAGILIM_BARAN_ALI.md` |
| Hocaya götürülecekler | `todos/HOCA_GUNDEM.md` |
| Per-WP detayları | `todos/WP-XX-*.md` |
| Paper draft | `LaTeX_DL_468198_240419/paper.tex` |
| Project conventions | `AGENTS.md` (root) |
