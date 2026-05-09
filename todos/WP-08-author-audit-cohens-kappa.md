# WP-08: 3-Judge Audit + Fleiss's κ (renamed from "Cohen's κ" — see D3)

**Owner:** **Joint** (Baran + Ali + TEDU external Hoca; Murat Hoca rater DEĞİL)
**Depends-on:** [WP-07 Judge LLM, all RQ runs (WP-03..06)]
**Effort:** M (~2-3 weeks total: 1-2 weeks recruitment + 1 hour calibration + ~12-15 hours per rater × 3)
**Status:** TODO
**Addresses:** [D3 lockdown, Hoca-6 (run/varyant/power), LLM Guidelines G5]
**Refs:** `MASTER_PLAN.md` §3 D3, `HOCA_GUNDEM.md` Konu 1

---

## Goal

LLM-Assisted Human Evaluation protocol'ün **insan tarafını sertleştir**: 3 bağımsız rater, ~150 stratified verdict, **Fleiss's κ** ile inter-rater agreement raporu.

**Why renamed**: Eski plan "Cohen's κ" diyordu (2-rater). D3 lockdown 3-rater (Baran + Ali + TEDU external) kararı verdi → Fleiss's κ doğru istatistik (Cohen's 2-rater için).

**Why critical**: ESE reviewer'ları "AI×AI evaluation reliable mi?" sorusuna **insanlı kanıt** istiyor. Bağımsız 3.-taraf (TEDU external) bu sorunun cevabıdır.

---

## Architecture

### 3 Rater Setup

| Rater | İlişki | Rolü |
|-------|--------|------|
| Baran | Author + rater | Internal rater 1 |
| Ali | Author + rater | Internal rater 2 |
| TEDU bağımsız Hoca | External (projeden bihaber) | **Independence anchor** |

**Murat Karakaya rater DEĞİL** — supervisor + yazar olarak kalır. Audit'a katılmaması "supervisor double-as-rater" methodolojik problemini önler.

### Sample Design (~150 verdict)

**Stratification**:
| Boyut | Quota | Total |
|-------|-------|-------|
| Violation type (V1-V6) | 25 verdict / type | 150 |
| Pipeline (P1/P2/P3) | dengeli (orthogonal) | — |
| Domain (D1/D2/D3) | dengeli (~50/domain) | — |
| Model | %50 G1, %50 OSS-mix | — |
| Judge confidence | %50 high-conf, %30 disagreement, %20 borderline | — |

**Sample selection**:
1. Aggregate all RQ run outputs (~1500-2000 violations total across all RQs)
2. Stratified random sample with quotas above
3. Anonymize: rater her case için sadece kod parçasını + SRS bağlamını + Judge'ın kararını görür; hangi pipeline/model olduğunu bilmez (blind audit)

### Rater Workflow

**Per case** (~5 dakika / case):
1. Read code snippet
2. Read SRS context (related domain rules)
3. Read Judge LLM's verdict (TP / FP / FN classification)
4. Decide: AGREE / DISAGREE / UNCERTAIN
5. Add 1-line comment if disagree

**Format**: Excel/Google Sheet, 1 row per verdict, 3 columns for ratings.

```
| case_id | code | srs_context | judge_verdict | baran_rating | ali_rating | tedu_rating | notes |
```

### Calibration Session (1 hour, pre-audit)

- 3 rater + framework author (Baran for protocol guidance) toplanır
- 5-7 örnek case üzerinden geçilir
- Edge case'lerde anlaşma (örn. "borderline V1 vs V3" — synonym mı naming convention mı?)
- Rubric'in muğlak yerleri netlenir
- **Pre-audit Fleiss's κ ≥ 0.70 hedef** (calibration cases üzerinde)

Eğer pre-audit κ < 0.70: rubric revize edilir, calibration tekrarlanır.

### Statistical Computation

**Primary metric**: Fleiss's κ (3-rater agreement)
- Computed across all 150 verdicts
- Per V-type κ also reported (V1, V2, ..., V6)

**Secondary metrics**:
- Pairwise Cohen's κ (Baran-Ali, Baran-TEDU, Ali-TEDU) — sub-analysis
- Krippendorff's α (more robust to missing data; sensitivity check)

**Implementation**: `core/eval/fleiss_kappa.py` (WP-17 ile coordineli)

```python
import numpy as np

def fleiss_kappa(ratings: np.ndarray) -> float:
    """
    ratings: shape (N_subjects, N_categories)
    Each row: count of raters who chose each category for that subject
    Returns: Fleiss's κ
    """
```

### Threshold (Landis-Koch)

| κ değeri | Yorum | Paper bar |
|----------|-------|-----------|
| < 0.20 | Slight | 🚨 Methodology krizi |
| 0.21-0.40 | Fair | 🟠 Yetersiz, rubric revize |
| 0.41-0.60 | Moderate | 🟡 ESE minimum, threats'ta disclose |
| **0.61-0.80** | **Substantial** | 🟢 **Hedef**: ESE rahatlıkla geçer |
| 0.81-1.00 | Almost perfect | 🟢 Mükemmel |

---

## Acceptance Criteria

- [ ] TEDU external Hoca recruited + onaylandı (`HOCA_GUNDEM.md` Konu 1 closed)
- [ ] Calibration session yapıldı (1 saat, 5-7 case, 3 rater)
- [ ] Pre-audit Fleiss's κ ≥ 0.70 (calibration cases üzerinde)
- [ ] 150 stratified verdict sample seçildi (`evaluation/audit_sample.csv`)
- [ ] 3 rater bağımsız olarak rate etti (`evaluation/audit_ratings_{baran,ali,tedu}.csv`)
- [ ] Fleiss's κ hesaplandı: overall + per V-type (`evaluation/kappa_results.json`)
- [ ] Pairwise Cohen's κ sub-analysis raporlandı
- [ ] **Hedef: Fleiss's κ ≥ 0.6 overall**; altıysa methodology paragrafında disclose
- [ ] Paper §4.5 ve §9.3 bu sonuçlarla integrate (WP-13'te)

---

## Implementation Steps

### W1-W3 — Recruitment + Calibration
1. (Baran) `HOCA_GUNDEM.md` Konu 1 — Murat Hoca courtesy bilgi + TEDU external Hoca'ya yaklaş
2. Onay alınınca: 1-sayfalık DDD-Enforcer brifingi hazırla (`evaluation/onboarding_brief.md`)
3. 5-7 calibration case seç (mevcut intermediate runs'tan)
4. Calibration session schedule

### W8 — Sample Design
5. RQ runs tamamlandığında (WP-03..06 done): aggregate all violations
6. `scripts/generate_audit_sample.py` — stratified random sample of 150
7. Anonymize + create rater spreadsheets

### W8-W11 — Audit Execution
8. 3 rater bağımsız olarak rate eder (deadline: 3 hafta)
9. Daily check-ins (kim ne kadar yaptı)

### W11 — Analysis
10. `scripts/compute_fleiss_kappa.py` — 3-rater matrix → κ
11. Per V-type breakdown
12. Pairwise Cohen's κ sub-analysis
13. Sonuçları `evaluation/kappa_results.json`'a yaz

### W11 — Integration with paper
14. WP-13 §4.5 ve §9.3'te κ sonuçlarını yaz

---

## Outputs

- `evaluation/onboarding_brief.md` (1-page intro for TEDU external)
- `evaluation/calibration_cases.json` (5-7 case)
- `evaluation/audit_sample.csv` (150 stratified verdicts)
- `evaluation/audit_ratings_baran.csv`
- `evaluation/audit_ratings_ali.csv`
- `evaluation/audit_ratings_tedu.csv`
- `evaluation/kappa_results.json`
- `scripts/generate_audit_sample.py`
- `scripts/compute_fleiss_kappa.py`
- `core/eval/fleiss_kappa.py`
- `tests/test_eval/test_fleiss_kappa.py`

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| TEDU external Hoca cevap vermez (>4 hafta) | Yüksek | Backup: Hocan başka aday önerir; veya 2-rater Cohen's κ + threats'ta disclose |
| Calibration κ < 0.7 — rubric ambiguous | Orta | Rubric revize, calibration tekrarla; max 2 iter |
| Audit sırasında κ < 0.6 (substantial altı) | Orta-yüksek | (a) rubric tighten, more cases discussed; (b) rapor olduğu gibi + threats'ta tartış |
| 150 verdict çok geliyor (12-15 saat / rater) | Düşük | Sample 100'e indir; güvenirlik az düşer ama kabul edilir |
| Anonymity bozulur (rater recognizes setup) | Düşük | Per-case randomized order; no model/pipeline labels visible |

---

## Sync Points

- **W8**: RQ runs done → Baran sample generation
- **W11**: Audit complete → Ali stats + Baran prose integration
