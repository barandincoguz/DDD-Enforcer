# Agent Quickstart — Read This First

> **Bu dosya, projeye yeni katılan bir kod ajanının (Claude Code, Cursor, vs.) tek geçişte projeyi anlaması için yazılmıştır. Sırayla oku.**
>
> **Last updated**: 2026-05-08 (post D1-D7 lock + audit restructure)

---

## 1. Proje 1 Paragrafta

**DDD-Enforcer**, mikroservis kodbazlarında **Domain-Driven Design (DDD)** ihlallerini tespit eden bir LLM-tabanlı framework. 4-aşamalı multi-agent pipeline (Scout → Architect → Specialist → Synthesizer) bir SRS dokümanından domain modelini çıkarır; AST-tabanlı validator + cross-family LLM-Judge bu modele karşı kodu denetler. **Hedef**: Springer **Empirical Software Engineering** (EMSE) regular track gönderim, ~14 hafta. **Yazarlar**: Baran Dincoguz + Ali Kendir + Prof. Dr. Murat Karakaya (supervisor).

Bu repo bir **konferans paper'ının (UBMK 2025) genişletilmiş dergi versiyonunu** üretiyor — paper draft `LaTeX_DL_468198_240419/paper.tex` içinde.

---

## 2. Mutlaka Oku (sırayla)

| Dosya | Okuma sırası | Ne öğrenirsin |
|-------|--------------|---------------|
| **`todos/AGENT_QUICKSTART.md`** (bu dosya) | 1 | Proje overview + entry points |
| **`todos/MASTER_PLAN.md`** | 2 | Locked decisions (D1-D7), 6-phase roadmap, 23 active WP, kritik path |
| **`todos/WP_DAGILIM_BARAN_ALI.md`** | 3 | Hangi WP kimde + sync points |
| **`todos/HOCA_GUNDEM.md`** | 4 | Hocaya götürülecek konular (validator, RQ5 drop, RQ1 pipeline count) |
| **`todos/INDEX.md`** | 5 | Hızlı status board (TODO/IN-PROGRESS/DONE) |
| **WP-XX-*.md dosyaları** | 6 | Sadece kendi sahip olduğun WP'leri detaylı oku |

**Senin kim olduğuna göre okuma yolu**:
- **Baran** → Master plan + WP-01a/01b/01c/01d/03/04/05/06/07/12/13/NEW-B
- **Ali** → Master plan + WP-02/10/11/14/16/17/NEW-A/NEW-C/NEW-D
- **Joint** → WP-00 + WP-08 + WP-15

---

## 3. Hemen Bilmen Gerekenler — Locked Decisions Özet

| # | Karar | Final |
|---|-------|-------|
| **D1** | **6 model** | G1: gemini-3.1-pro-preview · G2: gemini-3.1-flash-lite · O1: gpt-oss:120b-cloud · O2: qwen3-coder-next:cloud · O3: minimax-m2:cloud · O4: gemma4:31b-cloud |
| **D2** | **3 sektör + Yaklaşım F** | D1 e-ticaret (mevcut authored, publish) + D2 banking (public GitHub SRS) + D3 healthcare (public GitHub SRS). Her domain 3 codebase varyantı: CLEAN + DRIFT-LIGHT + DRIFT-HEAVY. Otomatik AST drift injector (V1-V6 quota) |
| **D3** | **3-rater Fleiss's κ** | Baran + Ali + 1 TEDU bağımsız Hoca. Murat Karakaya yazar+supervisor ama rater DEĞİL. ~150 stratified verdict |
| **D4** | **N=10 baseline** | Hafta 4 pilot (D1+P3+6 model, F1 std ölç) → variance düşükse N=5 |
| **D6** | **RQ5 silindi** | Reviewer'a bahsedilmez. Murat Hoca'ya courtesy bilgilendirme |
| **D7** | **Security false alarm** | .env hiç git geçmişinde olmadı |

**RQ yapısı (4 ana, 6 alt)**:
- RQ1: Pipeline (P1 vs P2 vs P3)
- RQ2a: Within-Gemini scaling (G1 vs G2)
- RQ2b: Closed vs OSS (best Gemini vs best OSS)
- RQ2c: OSS landscape (4 OSS karşılaştırma)
- RQ3: Cross-domain × 3 drift level
- RQ4: Synthetic seeded violations (V1-V6)

**Yeni metrik**: `json_failed` rate — Pydantic strict istendi ama model geçersiz JSON üretti mi?

---

## 4. 6-Phase Roadmap (yüksek seviye)

```
Phase 0 — Hijyen (DONE ✅)
Phase 1 — Çekirdek Altyapı (W2-4):  WP-00, 01a, 01b, 01c, 01d, NEW-B
Phase 2 — Veri + Eval (W3-5):       WP-02, NEW-A, 07
Phase 3 — RQ Yürütme (W5-9):        WP-03, 04, 05, 06
Phase 4 — Validation + Stats (W8-11): WP-08, 17, NEW-C
Phase 5 — Yazım (W10-13):           WP-10, 11, 13, 14, 16
Phase 6 — Replication + Submit (W13-15): WP-12, NEW-D, 15
Phase 7 — Buffer (W15-16)
```

**Critical path**: P0 (DONE) → WP-01a → WP-01b → WP-02 → WP-NEW-A → WP-03/04/05/06 → WP-08 → WP-13 → WP-12 → WP-15

**4 critical blocker** (audit'ten):
1. WP-NEW-A (AST drift injector) — yoksa Faz 2 chokepoint
2. WP-08 3-Judge setup — TEDU external Hoca recruitment Hafta 1'de başlamalı
3. WP-17 N pilot gate (Hafta 4) — RQ batch'leri yanlış N ile koşmasın
4. RQ5 cleanup — INDEX.md ve WP-13'te artık WP-18 referansı olmamalı

---

## 5. Conventions — Bunları İhlal Etme

### Code Style
- **Provider-agnostic LLM client** mimarisi: `core/llm/` paketi (base.py + errors.py + retry.py + registry.py + gemini.py + ollama.py + schema_probe.py). Eski `core/llm_client.py` **silinecek** (big-bang refactor)
- **Pydantic strict schemas** her LLM çıktısında zorunlu
- **Type hints** her public function'da
- **No global state** dışında: TokenTracker, ValidationMetricsTracker (mevcut singleton'lar, korunur)

### Test
- **TDD zorunlu**: Test önce yazılır, kırmızı geçer, sonra implementation
- **Integration tests** `@pytest.mark.integration` ile işaretlenir; CI'da skip edilir (live API gereksiz)
- **Coverage gate**: %60 minimum (mevcut %67); WP eklediğin yerde testler de eklenmeli
- **conftest.py** API key placeholder set eder (CI test için)

### Commit
- **Atomic commits**: Her commit yeşil-CI olmalı, rollback mümkün olsun
- **Convention**: `<type>(<scope>): <description>` (örn. `feat(llm): add OllamaClient with key rotation`)
- **Co-author trailer**: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` (Claude Code kullanıyorsan)
- **Big-bang refactor scope**: WP-01a eski client'ı tek tip silmek demek; AMA 9 atomic commit'e bölünmüş, her biri ara yeşil state

### Branch
- `main` her zaman yeşil; feature work `wp-XX-*` veya `feature/wp-XX-*` branch'inde

### File Ownership
- WP_DAGILIM_BARAN_ALI.md'ye bak, sahibin olduğu WP'ler dışındaki dosyalara dokunma
- Ortak alan: `paper.tex`, `LaTeX_DL_*/`, `extension/backend/configs/` — koordineli edit, PR review zorunlu

### Sync Points (W4, W5, W7, W9, W10-11, W11, W13)
- `todos/HANDOFF_S{1..7}.md` template'leriyle teslim edersin (gelecekte oluşturulur)
- Read-only data handoff (CSV, manifest, schema). Concurrent code edit yok

---

## 6. Sıradaki Concrete Action

**Sahibine göre**:

### Baran ise
1. **WP-01a TDD Commit-1**: `core/llm/base.py` + `core/llm/errors.py` + `tests/test_llm/test_base.py` + `tests/test_llm/test_errors.py`. ~8 test, hepsi yeşil. Detay: `todos/WP-01a-provider-abstraction.md`.
2. Sonrası: 9-commit sequence (base → registry → retry → ollama → gemini → architect-migration → main-migration → delete-old → schema-probe).

### Ali ise
1. **WP-NEW-A**: AST Drift Injector smoke prototype. `scripts/inject_drift.py`. Pure Python AST manipülasyon, LLM gerek yok. Detay: `todos/WP-NEW-A-ast-drift-injector.md`.
2. **Paralel olarak WP-02a**: D2 (banking) public SRS sourcing — GitHub `software-requirements-specification` topic'inde aday tara, license check.
3. Hafta 4'e kadar D1 publish + `subjects/D1/code-clean/` Gemini Pro ile generate.

### Hoca konuşması (her iki yazar da bilmeli)
- `todos/HOCA_GUNDEM.md`'deki 3 konu (validator request, RQ5 drop, RQ1 pipeline count) bu hafta Hocayla konuşulmalı

---

## 7. Çalışma Ortamı (Quick Setup)

```bash
# Repo
git clone https://github.com/barandincoguz/DDD-Enforcer.git
cd DDD-Enforcer

# Python (3.12 pinned)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r extension/backend/requirements.lock  # hash-pinned, reproducible

# Local .env (sen oluştur, commit edilmez — .gitignore'da)
echo "GEMINI_API_KEY=<your-key>" > extension/backend/.env
echo "OLLAMA_API_KEYS=key1,key2,key3,key4,key5,key6" >> extension/backend/.env

# Tests
cd extension/backend
pytest -m "not integration"  # ~5 saniye, 105 test geçer
```

**CI**: GitHub Actions `.github/workflows/backend-ci.yml` — Python 3.12, pytest + coverage (≥%60), pyright (continue-on-error), strict markers.

---

## 8. Kritik Dosya Konumları

| Konum | İçerik |
|-------|--------|
| `extension/backend/core/` | Backend Python kodu (architect.py, llm_client.py-OLD, validation.py, AST/, code_parser/) |
| `extension/backend/tests/` | pytest test suite (105 unit + 29 integration) |
| `extension/backend/configs/` | models.py (registry), scope.yaml (TBC) |
| `extension/backend/inputs/` | D1 SRS.docx |
| `extension/backend/core/intermediate/` | Pipeline aşama çıktıları (Scout/Architect/Specialist/Synthesizer JSON dumps) |
| `LaTeX_DL_468198_240419/` | Paper draft (paper.tex), Springer template |
| `resources/` | Referans PDF'ler (mevcut DDD literatürü) |
| `todos/` | Tüm planning artifact'ları (bu dosya, MASTER_PLAN, WP-XX'ler) |
| `subjects/` (TBC) | D1/D2/D3 SRS + codebase variants — WP-02'de oluşturulacak |
| `runs/` (TBC) | Run-spec YAML'ları + output JSON'ları — WP-01b'de oluşturulacak |

---

## 9. Sorularım Olursa

- **Dokuman çelişkisi**: MASTER_PLAN.md (woolly-hopping-moonbeam.md ile aynı) **canonical**'dır. WP-XX dosyaları onunla çelişirse MASTER_PLAN kazanır
- **WP'm kapsam dışında bir şey gerektiriyor**: WP_DAGILIM'da kontrol et, başkasının alanına giriyorsa sahibine yaz
- **Mimari kararı değiştirmek istiyorum**: HOCA_GUNDEM'e yeni konu ekle veya master plan'a "değişiklik teklifi" diff'i öner
- **Sync point öncesi handoff hazırlamak**: `todos/HANDOFF_S{N}.md` template (gelecekte örnekler eklenecek)

---

**Sıradaki adım: `MASTER_PLAN.md`'yi oku.**
