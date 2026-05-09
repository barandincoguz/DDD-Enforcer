# Work-Package Dağılımı — Baran ↔ Ali

> **Amaç**: Communication overhead'i minimize ederek, birbirini etkileyen WP'leri tek elde toplamak. Read-only data handoff'ları (CSV, manifest) sync point'lerde gerçekleşir.
>
> **Toplam aktif WP**: 23 (Baran 11 + Ali 10 + Joint 2)
>
> **3. Yazar**: Murat Karakaya (supervisor) — yazar, ama WP execution'a katılmıyor; review gates'inde.

---

## Allocation Principle

İki kişiyi paralel çalıştırmanın altın kuralı: **"birbirini etkileyen WP'ler aynı kişide olur"**. Bir WP başkasının kodunu/datasını "sürekli değiştirir" niteliğindeyse → tek el. "Bir kez okuyup işlerine bakar" niteliğindeyse → split edilebilir, sync point'te handoff yapılır.

**Cohesion clusters** (audit sonrası):

| Cluster | WP'ler | Coupling tipi |
|---------|--------|---------------|
| **α LLM Infrastructure** | 01a/b/c/d, NEW-B, 07, NEW-C | Tüm WP'ler `core/llm/` paketini kullanır — change cascade riski |
| **β Data + Drift** | 02, NEW-A, 06 | Drift injector → corpus → RQ4 zinciri; AST manipülasyonu paylaşılır |
| **γ RQ Chain** | 03 → 04 → 05/06 | "Winner passes forward" + `runs/` orchestrator schema |
| **δ Stats + Audit** | 17, NEW-C, 08 (joint) | RQ output'larını tüketir; istatistik script'leri |
| **ε Writing** | 10, 11, 13, 14, 16 | Gevşek bağlı, parallelize edilebilir |
| **ζ Submission** | 12, NEW-D, 15 | Final paketleme |

**En kritik karar**: α ve γ **aynı kişide**. γ (RQ runs) doğrudan α (LLM clients) kullanıyor. Ayrı kişilerde olursa runtime sorunlarında ("model X retry config?") sürekli sync gerekir.

---

## Sahiplik Tablosu

### 🔵 BARAN — 11 WP

**Faz 1 — Çekirdek Altyapı (5 WP, hepsi α cluster)**
- **WP-01a** Provider abstraction (`core/llm/` paketi, big-bang refactor)
- **WP-01b** Run orchestrator (idempotent worker, run-spec YAML)
- **WP-01c** Token tracking + json-failed metric integration
- **WP-01d** P1/P2/P3 pipeline classes
- **WP-NEW-B** Schema-conformance probe (6 model × 3 schema smoke)

**Faz 2 — Eval Altyapı (1 WP)**
- **WP-07** Judge LLM rubric pipeline (cross-family Judge)

**Faz 3 — RQ Yürütme (4 WP, γ cluster)**
- **WP-03** RQ1 execution
- **WP-04** RQ2 execution (a/b/c sub-RQ'lar)
- **WP-05** RQ3 cross-domain
- **WP-06** RQ4 synthetic violations *(corpus Ali'den, runtime Baran'da)*

**Faz 5 — Yazım (1 WP)**
- **WP-13** Discussion + threats prose (RQ sonuçlarını en iyi bilenin elinde)

**Faz 6 — Replication (1 WP)**
- **WP-12** Replication package + Zenodo DOI

**Toplam**: 11 WP. **α + γ + ζ-partial bundle** = code path tek elde.

### 🟢 ALI — 10 WP

**Faz 2 — Veri (3 WP, β cluster)**
- **WP-02** Subject corpus (sourcing + clean codebase generation + drift variants)
  - 02a: D2/D3 public SRS sourcing (GitHub topic search + license)
  - 02b: D1 publish + 3 domain CLEAN codebase üretimi (G1 Gemini Pro ile)
  - 02c: DRIFT-LIGHT/HEAVY varyantları (NEW-A çıktısıyla)
- **WP-NEW-A** AST drift injector (V1-V6 quota tool — pure Python, LLM gerek yok)

**Faz 4 — Stats + Ablation (2 WP, δ cluster)**
- **WP-17** Statistical methodology (Wilcoxon, Friedman+Nemenyi, Cliff's δ, Fleiss's κ implementation, power analysis, pilot gate)
- **WP-NEW-C** Prompt sensitivity ablation (3 prompt variant × pipeline)

**Faz 5 — Yazım (4 WP, ε cluster çoğu)**
- **WP-10** Bibliography cleanup (24 ref verify, arXiv ID forensics, 8+ yeni ref)
- **WP-11** Figures (4 vector PDF: architecture, RQ2 6-model Pareto, RQ4 recall, extension sequence)
- **WP-14** Abstract + conclusion polish (numerically grounded, WP-13 sonrası)
- **WP-16** Extension architecture documentation (1.5 sayfa + 3-5 screenshot)

**Faz 6 — Submission (1 WP)**
- **WP-NEW-D** Reviewer/Evaluator Guide (`EVALUATION.md`, 30-dk artifact-eval recipe)

**Toplam**: 10 WP. **β + δ + ε + ζ-partial bundle** = data + analysis + writing + reviewer-facing materyaller.

### 🟡 JOINT — 2 WP + WP-08

- **WP-00** Scope lock (30 dk kickoff, beraber yazılır → tek commit)
- **WP-15** Submission package (cover letter + EM portal final assembly — son gün, beraber)
- **WP-08** 3-Judge audit + Fleiss's κ — **doğası gereği joint**: Baran rate eder, Ali rate eder, TEDU external rate eder, sonuçlar agrege edilir. Calibration session da 3 rater + sen + Ali

---

## Sync Points (Critical Handoffs)

Her sync point'te bir READ-ONLY transfer var (CSV, manifest, schema). Aynı kodda concurrent edit yok.

| # | Hafta | Olay | Veren | Alan | Format |
|---|-------|------|-------|------|--------|
| **S1** | W4 sonu | LLM clients stable | Baran (WP-01a green) | Ali | API contract docs (`core/llm/__init__.py` exports) — Ali codebase generation için kullanır |
| **S2** | W5 sonu | Subject corpora ready | Ali (WP-02 done) | Baran | `subjects/D{1,2,3}/{clean,drift-light,drift-heavy}/` directories — Baran RQ runs başlatır |
| **S3** | W7 ortası | Pilot variance results | Baran (Pilot Hafta 4) | Ali (decision) | F1 std table — Ali N=5 vs N=10 kararını verir, WP-17 SAP'a yazar |
| **S4** | W9 sonu | RQ outputs complete | Baran (WP-03/04/05/06 done) | Ali | `runs/outputs/*.json` — Ali stats hesaplar (WP-17), figürler oluşturur (WP-11) |
| **S5** | W10-11 | Audit calibration + execution | **JOINT** (Baran + Ali + TEDU external) | Both | Calibration agenda + 150 stratified verdict Excel |
| **S6** | W11 sonu | Stats results | Ali (WP-17 done) | Baran | Statistical findings document — Baran WP-13 discussion'a integrate eder |
| **S7** | W13 sonu | Writing complete | Both done | Joint | All sections written → WP-12 (Baran) + WP-15 (joint) assembly |

---

## Effort Balance Check

| Phase | Baran | Ali | Joint |
|-------|-------|-----|-------|
| Faz 1 (W2-4) | 5 WP infra-heavy | WP-02 sourcing başlar (paralel), NEW-A başlar | WP-00 kickoff |
| Faz 2 (W3-5) | WP-NEW-B + WP-07 | WP-02 codebase generation, NEW-A finalize | — |
| Faz 3 (W5-9) | RQ chain (4 WP) — yoğun | WP-17 framework, NEW-C prompt sensitivity | — |
| Faz 4 (W8-11) | WP-13 outline | WP-17 stats execution | **WP-08 audit** (W10-11) |
| Faz 5 (W10-13) | WP-13 derinleşir | WP-10, 11, 14, 16 paralel | — |
| Faz 6 (W13-15) | WP-12, NEW-D | (review pass) | **WP-15 final assembly** |

**Ali gevşek dönemler**: W2-4 ilk yarısı (Baran infra'yı yetiştirirken Ali sadece WP-02a sourcing + NEW-A pure-AST işi yapar). Kompansasyon: W10-13 yazım yoğunluğunda 4 WP (10, 11, 14, 16) Ali'de.

**Baran yoğun dönemler**: W5-9 RQ chain. W2-4 LLM altyapısı.

Effort tahmin (rough M/L/S):
- Baran: 2L + 6M + 3S ≈ 38-46 person-days  
- Ali: 1L + 5M + 4S ≈ 32-40 person-days

Ali biraz daha hafif (~%15), bu mantıklı çünkü Baran kritik path üzerinde. Eğer eşitlemek istersek WP-NEW-D'yi (Reviewer Guide) Ali'ye verebiliriz — ama o zaman Ali Baran'ın replication paketini "tüketmek" yerine ortak şekillendirir, sync point ekler. Mevcut allocation iyi.

---

## Risk + Contingency Matrisi

| Senaryo | Etki | Mitigation |
|---------|------|-----------|
| **WP-02 (Ali) D2/D3 sourcing 2 hafta gecikir** | RQ3, RQ4 D2/D3 üzerinde geç başlar | Baran D1-only RQ1/RQ2'yi başlatır (D1 hazır), D2/D3 W7'de eklenir |
| **WP-NEW-A (Ali) drift injector çalışmaz** | Yaklaşım F kırılır, sadece CLEAN codebase'lerle iş yaparız | Fallback: manuel drift hand-edit (WP-06 eski plan), ama agresif zaman kaybı (~1 hafta) |
| **WP-01a (Baran) refactor planlanandan uzun sürer** | RQ chain başlangıcı kayar | Faz 1.5 tamponu: Hafta 5'te eski client geçici kullan, refactor paralel devam |
| **TEDU external Hoca cevap vermez** | WP-08 Fleiss's κ kırılır | Backup: 2-rater Cohen's κ (Baran + Ali), threats-to-validity'de "no external" disclose |
| **Ali pilot sonucu istenmez (variance > 0.15)** | N=10 → N=20'ye escalation, compute 2× | Bütçe rahat ($300 kredi), sadece zaman ek (~1 hafta) |
| **Baran RQ runs Cloud session limit'e takılır** | RQ batch'leri yarıda kalır | Idempotent worker (WP-01b) zaten resume özelliği sağlar; Pro $20 abonelik fallback |

---

## Communication Protocol

**Daily standup (5 dk async)**: Slack/WhatsApp/Discord — herkesin günün sonunda 3 satır:
- Ne yaptım?
- Yarın ne yapacağım?
- Engelim var mı?

**Weekly sync (30-45 dk)**: Pazartesi sabahı 30-45 dk Zoom/teams. Sync point yaklaşıyorsa veya engel varsa.

**Sync point hand-off**: Sahibi handoff document yazar:
- Format: `todos/HANDOFF_S{1,2,...}.md`
- İçerik: ne hazır, nasıl tüketilecek (örnek komut), bilinen sorunlar, schema versiyonu
- Alıcı handoff'u kabul eder veya soru sorar

**Code review**: Her commit'i karşı taraf review eder (PR-based veya post-commit review). TDD nedeniyle test'ler var, review hızlı geçer.

**Sahip değişikliği gerekirse**: GitHub'da `assignee` re-assign + daily standup'ta açıkla. WP'yi geri alabilir başkasına verirsin.

---

## Sonraki Adım

Bu dağılımı Ali ile paylaş. Ali kabul ederse veya değişiklik isterse:
- Bu dosya güncellenir
- HOCA_GUNDEM.md'ye Konu 4 olarak kısa bilgilendirme eklenir (Hoca da kim ne yapıyor görsün)

Onaylanırsa, sıradaki concrete adım: **WP-01a TDD Commit-1** (Baran'ın işi).

---

## Değişiklik Geçmişi
- **2026-05-08 v1**: İlk dağılım, audit sonrası 23-WP üzerine kuruldu
