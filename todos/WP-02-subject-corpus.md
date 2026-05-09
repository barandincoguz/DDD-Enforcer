# WP-02: Subject Corpus (3 Domain × 3 Codebase Variant + Public SRS Sourcing)

**Owner:** Ali
**Depends-on:** [WP-00 scope, WP-NEW-A drift injector for sub-task 02c]
**Effort:** L (~2.5-3 weeks; longest single WP)
**Status:** TODO
**Addresses:** [D2 Yaklaşım F, Hoca-1 K=3 domains]
**Refs:** `MASTER_PLAN.md` §3 D2

---

## Goal

3 farklı sektörden 3 SRS + her biri için 3 codebase varyantı (CLEAN, DRIFT-LIGHT, DRIFT-HEAVY) üret. Yaklaşım F'in datayı oluşturma adımı. Reviewer için "controlled experimental" çerçeve.

**Critical**: SRS'ler **public GitHub kaynaklarından** sourced olacak (D2/D3 için). D1 mevcut authored, repo'ya publish edilecek. Bu provenance "researcher bias" itirazını ciddi azaltır.

---

## Sub-Tasks

### 02a — D2 + D3 Public SRS Sourcing (W3, ~7 gün)

**Goal**: GitHub topic `software-requirements-specification` veya benzeri public repolardan banking + healthcare SRS bul, license check yap, repo'ya kopyala.

**Pre-screen kriterleri**:
- Uzunluk: 3000-8000 kelime
- DDD-friendly vocabulary: business entities, rules, glossary kısmı var mı? (use-case-only → reddet)
- License: MIT, Apache, CC BY, public-domain — verify
- İngilizce
- 4-7 bounded context çıkma potansiyeli

**Aday havuzu hedefi**: ~10 SRS shortlist → ~3-5 detaylı inceleme → 3 final seçim

**Adım adım**:
1. GitHub: `topic:software-requirements-specification language:markdown OR language:tex` arama
2. GitHub: `topic:srs-document` ve `topic:srs` topic'lerini de tara
3. **Banking adayları**: BankaProgram-SRS, Bank-Management-System-SRS, Online-Banking-System variantları, Volere case studies (banking örneği var)
4. **Healthcare adayları**: Hospital-Management-System SRSes, Medical-IoT-System, Pharmacy-Management-System
5. Her aday için: license dosyasını kontrol et, içerik DDD-friendly mi review et, kelime sayısını ölç
6. Top 3 banking + top 3 healthcare adayı **shortlist'e koy** → kısa README ile `subjects/_candidates/D2-banking/` ve `subjects/_candidates/D3-healthcare/` altına klonla
7. Final 1 banking + 1 healthcare seç (Baran ile birlikte review), `subjects/D2/srs.md` ve `subjects/D3/srs.md` olarak commit et

**Acceptance**:
- [ ] `subjects/_candidates/` altında 6+ SRS adayı (3 banking + 3 healthcare) klonlanmış
- [ ] License dosyası her birinde mevcut, MIT/Apache/CC-BY/public
- [ ] Final D2 ve D3 SRS dosyaları `subjects/D2/srs.md` ve `subjects/D3/srs.md` olarak commit edilmiş
- [ ] `subjects/_sourcing_log.md`: hangi repo'dan, hangi commit hash, hangi license, neden seçildi (ya da reddedildi)

**Output**:
- `subjects/D2/srs.md`
- `subjects/D3/srs.md`
- `subjects/_sourcing_log.md`

### 02b — D1 Publish + 3 Domain CLEAN Codebase Generation (W4, ~7 gün)

**Goal**: D1 mevcut authored SRS'i repo'ya formal publish et + 3 domain için **CLEAN** Python microservice codebase üret (Gemini Pro ile generate, hand-edit minimal).

**Adım adım**:
1. **D1 publish**: `extension/backend/inputs/SRS.docx` → `subjects/D1/srs.md` (markdown convert, license: "Authored by Dincoguz/Kendir for this study, CC BY 4.0")
2. **CLEAN codebase generation prompt template** hazırla:
   ```
   Generate a Python microservice codebase implementing the following SRS.
   Use FastAPI for services, Pydantic for domain models, and clear bounded
   context separation. Aim for 4-7 services, 500-2000 LOC total.
   No external infrastructure (databases, message brokers) — use in-memory.
   Code should be DDD-textbook clean: clear ubiquitous language, no banned
   terms, proper aggregate boundaries.
   
   SRS:
   {srs_content}
   ```
3. **Generation runs**: Gemini 3.1 Pro Preview ile 3 SRS için 1'er run yap. Her run sonrası:
   - Output'u `subjects/D{1,2,3}/code-clean/` altına yaz
   - `cloc` ile LOC ölç, `subjects/D{N}/manifest.json`'a kaydet (file count, LOC, services)
   - Smoke test: `python -m mypy subjects/D{N}/code-clean/` (type check)
   - Smoke test: tüm dosyalar parse edilebiliyor mu (`ast.parse`)
4. **Manuel hand-edit (light)**: Generation'da gözden kaçan hataları düzelt (import errors, obvious typos). **DDD ihlali eklemiyoruz** — sadece çalışır hâle getiriyoruz. Her edit'i commit message'da belirt.

**Acceptance**:
- [ ] `subjects/D{1,2,3}/code-clean/` altında 4-7 mikroservis dizini, 500-2000 LOC
- [ ] Her dizin `mypy --strict` geçer (ya da en azından parse edilir)
- [ ] `subjects/D{N}/manifest.json`: services, LOC, files, generation_model, generation_date
- [ ] Hand-edit log'u commit messagelar'da

**Output**:
- `subjects/D1/srs.md`, `subjects/D1/code-clean/`, `subjects/D1/manifest.json`
- `subjects/D2/code-clean/`, `subjects/D2/manifest.json`
- `subjects/D3/code-clean/`, `subjects/D3/manifest.json`

### 02c — DRIFT-LIGHT + DRIFT-HEAVY Variants (W5, ~4 gün — depends on WP-NEW-A)

**Goal**: WP-NEW-A AST drift injector ile 3 domain × 2 drift-level = 6 codebase variant üret.

**Drift quotas** (V1-V6 dengelenmiş):
- DRIFT-LIGHT: 3-5 violations / domain. Quota: V1=1, V2=1, V3=1, V4=1, V5=0-1, V6=0-1
- DRIFT-HEAVY: 10-15 violations / domain. Quota: V1=2-3, V2=2-3, V3=2-3, V4=2, V5=1-2, V6=1-2

**Adım adım**:
1. WP-NEW-A drift injector ready (Ali'nin kendi WP'si)
2. Her CLEAN codebase'i input olarak ver, DRIFT-LIGHT manifest config'i ile çalıştır
3. Output: `subjects/D{N}/code-drift-light/` + `subjects/D{N}/code-drift-light/_drift_manifest.json`
4. Aynı işlemi DRIFT-HEAVY için tekrarla
5. Manuel review: random 5 ihlal seç, kontrol et — gerçekçi mi, V-type doğru mu?
6. Acceptance kontrolü: drift counts manifest'lerle eşleşiyor mu, codebase hâlâ parse ediliyor mu

**Acceptance**:
- [ ] `subjects/D{1,2,3}/code-drift-light/` 3 dizin
- [ ] `subjects/D{1,2,3}/code-drift-heavy/` 3 dizin
- [ ] Her drift dizininde `_drift_manifest.json`: hangi dosyada hangi V-type ihlali var, original→edited diff
- [ ] Manuel spot-check: 5/5 random ihlal "natural" görünüyor

**Output**:
- 6 drift codebase dizini + 6 manifest dosyası

---

## Önemli Decisions (Locked)

- **Codebase generation model**: Gemini 3.1 Pro Preview (G1)
- **Generation strategy**: 1 run per domain, hand-edit minimum
- **Target size**: 500-2000 LOC, 4-7 services
- **License**: D1 CC BY 4.0 (authored), D2/D3 inherit source license
- **No databases**: In-memory only (test simplicity)

---

## Sync Points

- **End of 02a (~W4 day 5)**: D2/D3 SRS'leri seçildi → Baran review
- **End of 02b (~W5)**: CLEAN codebase'ler hazır → WP-NEW-A drift injector input olarak kullanır
- **End of 02c (~W5 sonu)**: **S2 handoff to Baran** — `subjects/D{1,2,3}/{clean,drift-light,drift-heavy}/` → Baran RQ runs için kullanır

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Public SRS adayları kalitesiz, hiçbiri DDD-friendly değil | Orta | Volere library/banking case'i fallback (akademik host) |
| LLM-generated codebase çok "clean" — gerçekçi değil | Orta | Yaklaşım F drift injector ihlali ekler; hand-edit "natural feel" sağlar |
| Codebase generation çok uzun sürer (LLM session limit) | Düşük | 3 SRS × 1 run × ~5 dk = 15 dk; sorun yok |
| D2/D3 SRS license uyumsuzluk | Orta | License check 02a'da zorunlu; uyumsuz aday reject |
| Drift injector (WP-NEW-A) gecikirse 02c blocked | Yüksek | Ali her ikisini de paralel ilerletir; NEW-A pure-AST tool, dış bağımlılık yok |
| Hand-edit DDD-pure dengesini bozar (SRS bias) | Orta | Hand-edit log'u kayıt; sadece syntactic fixes (import, typo) |

---

## Communication Protocol

- Daily standup: hangi SRS adayını incelediğin, ne kadar yakın seçtiğin
- Sync 02a end: Baran ile 30 dk top-3 review meeting
- Sync 02c end: drift manifest'leri Baran review

---

## Cross-References

- D1 mevcut SRS: `extension/backend/inputs/SRS.docx`
- Drift injector: `WP-NEW-A-ast-drift-injector.md`
- DDD violation taxonomy V1-V6: `paper.tex` §3.5 (or `docs/violation_taxonomy.md` if extracted)
- License check guidance: `subjects/_sourcing_log.md` (template)
