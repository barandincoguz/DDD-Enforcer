# WP-04: RQ2 Model Comparison (Split into RQ2a / RQ2b / RQ2c)

**Owner:** Baran
**Depends-on:** [WP-03 winner pipeline, WP-01a 6-model client, WP-02 codebases, WP-07 Judge]
**Effort:** M (~2 weeks)
**Status:** TODO
**Addresses:** [D1 6-model lockdown, Hoca-1 RQ2 enabler]
**Refs:** `MASTER_PLAN.md` §4 RQ2 split

---

## Goal

RQ2'yi **3 alt-RQ'ya bölerek** koş ve raporla. Eski plan'da tek "model comparison" idi; D1 lockdown ile **3 sorulu yapı**a geçti:

- **RQ2a**: Within-Gemini scaling (G1 vs G2)
- **RQ2b**: Closed vs OSS frontier (best Gemini vs best OSS)
- **RQ2c**: OSS landscape (4 OSS karşılaştırma)

**Yeni metrik**: Tüm RQ2 alt-tablolarına `json_failed_rate` kolonu eklenir.

---

## RQ2a — Within-Gemini Scaling

**Soru**: "DDD-Enforcer için Gemini Pro Preview gerekli mi, Flash-Lite yeterli mi?"

**Tasarım**:
- Modeller: G1 (gemini-3.1-pro-preview) + G2 (gemini-3.1-flash-lite)
- Pipeline: WP-03'ten kazanan (büyük olasılıkla P3 multi-agent)
- Domain: D1 (CLEAN + DRIFT-LIGHT + DRIFT-HEAVY)
- N=10 run / cell
- Total: 2 × 1 × 3 × 10 = 60 runs

**Metrikler**:
- Precision (CLEAN'de) → "false alarm yokluğu"
- Recall (DRIFT-L/H'de) → "gerçek ihlal yakalama"
- F1 (combined)
- json_failed_rate
- Latency, cost per run

**İstatistik**:
- Wilcoxon signed-rank (paired: Pro vs Flash-Lite per same input)
- Effect size: Cliff's δ
- 95% CI bootstrap

**Beklenen sonuç**: Pro ≥ Flash-Lite, ama fark anlamsız ise "Flash-Lite production'da yeterli" diyebiliriz (çok güçlü pratik finding).

---

## RQ2b — Closed vs OSS Frontier

**Soru**: "En iyi closed (G1) vs en iyi OSS — fark anlamlı mı? OSS'un sıfır marjinal-maliyeti telafi ediyor mu?"

**Tasarım**:
- Closed: G1 (RQ2a winner — büyük olasılıkla Pro Preview)
- OSS: RQ2c'den çıkan en iyi OSS
- Pipeline: WP-03 winner
- Domain: D1
- N=10
- Total: 2 × 1 × 3 × 10 = 60 runs

**Metrikler**: Aynı RQ2a'yla, plus **cost-quality Pareto** vurgusu (closed pahalı vs OSS bedava).

**İstatistik**:
- Mann-Whitney U (independent, cross-family)
- Effect size: Cliff's δ
- Practical significance threshold: 5pp F1 ya da %20 cost saving

---

## RQ2c — OSS Landscape (4 OSS karşılaştırma)

**Soru**: "OSS arasında code-specialization mı, generalist reasoning mı, frontier-MoE mi öne çıkıyor?"

**Tasarım**:
- Modeller: O1 (gpt-oss:120b), O2 (qwen3-coder-next), O3 (minimax-m2), O4 (gemma4:31b)
- Pipeline: WP-03 winner
- Domain: D1
- N=10
- Total: 4 × 1 × 3 × 10 = 120 runs

**Metrikler**: Aynı + json_failed_rate vurgusu (özellikle gemma4 ve qwen için).

**İstatistik**:
- **Friedman test** (multi-model, paired)
- Posthoc: **Nemenyi test** (which OSS is significantly better)
- Holm-Bonferroni correction (multiple comparison)
- Cliff's δ pairwise

**Beklenen sonuç**: Code-specialized (qwen3-coder, minimax-m2) > generalist (gpt-oss, gemma4)? Veya MoE (minimax-m2) > dense (qwen3-coder)? Empirik olarak öğreneceğiz.

---

## Run Plan

Total RQ2 runs: 60 (2a) + 60 (2b, partial overlap with 2a) + 120 (2c) = ~200 runs.

Effective: ~180 unique runs (overlap minimization). Cost: ~$70 (Gemini) + ~$0 (OSS).

Time: ~30 saat compute (paralelizasyon ile ~12 saat wall clock).

---

## Acceptance Criteria

- [ ] RQ2a, RQ2b, RQ2c için ayrı sonuç tabloları (`runs/outputs/rq2a/...`, `runs/outputs/rq2b/...`, `runs/outputs/rq2c/...`)
- [ ] Her tabloda: P, R, F1, json_failed_rate, latency, cost
- [ ] İstatistiksel test sonuçları: p-values, effect sizes, CIs
- [ ] Paper §6 alt-bölümleri: §6.1 RQ2a, §6.2 RQ2b, §6.3 RQ2c (her birinde Tablo + analiz prose)
- [ ] **Pareto frontier** plot (Figure 2): F1 vs cost across all 6 models
- [ ] Discussion section'da "kazanan kategori" yorumu

---

## Implementation Steps

1. **Pre-req check**: WP-03 tamamlandı (winning pipeline locked), WP-01a tüm 6 model client çalışıyor
2. **Run-spec generation**: `scripts/generate_run_specs.py --rq=rq2a` etc. — generate YAML run-specs
3. **Execute runs**: `scripts/run_worker.py` — idempotent worker tüm cell'leri tek tek doldurur
4. **Aggregate results**: `scripts/build_rq2_tables.py` — runs/outputs'tan tablo render
5. **Statistical analysis**: WP-17 stat scripts'leri (Wilcoxon, Friedman, Nemenyi, Cliff's δ)
6. **Figure**: `scripts/plot_pareto_rq2.py` — vector PDF Figure 2
7. **Prose**: WP-13 ile birlikte §6.1, §6.2, §6.3 yazılır

---

## Outputs

- `runs/outputs/rq2{a,b,c}/...` (200 koşum dosyası)
- `runs/tables/rq2{a,b,c}.csv` (rendered tables)
- `runs/figures/fig2_pareto.pdf`
- Paper §6 alt-bölümleri (WP-13'te yazılır)

---

## Risks

| Risk | Mitigation |
|------|------------|
| OSS modellerden biri schema sürekli kayar (json_failed > 50%) | WP-NEW-B probe'da yakalanır; o model "Tier 2" olarak rapor edilir, kazanan seçimi etkilenmez |
| Friedman test 4 model için sığ — Nemenyi posthoc düşük güç | N=10 bunu kompanse eder; gerekirse N=20'ye escalation |
| Cost > $300 budget | OSS bedava, Gemini ~$70 estimated; çok altında, marj rahat |
| Pilot variance >0.15 → N=20 escalation | Compute 2× artar (~24 saat wall clock); kabul edilebilir |
