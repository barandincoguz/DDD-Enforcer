# Literature Search — EMSE Journal Extension

**Date:** 2026-04-27
**Author:** Drafted from Claude (Opus 4.7) training-data recall (cutoff Jan 2026). Web search/fetch were unavailable during this session, so **every entry below is tagged with a verification status**. Baran/Ali must verify each citation against Google Scholar / arXiv / Crossref before BibTeX migration in **WP-10**. Treat this file as a *shortlist with provenance*, not a finished bibliography.

## Verification tags

- 🟢 **VERIFIED-FROM-MEMORY** — Well-known, high-confidence; reasonably citable from memory but still verify DOI/page numbers before submit.
- 🟡 **PROBABLE** — Likely real, but exact metadata (authors, year, venue) may be slightly off; verify before submit.
- 🔴 **RISKY** — Uncertain provenance; treat as a search seed, not a citation. Confirm or replace.

**Rule:** Any 🔴 entry that survives into the final BibTeX must have its full bibliographic metadata re-checked against the publisher's site. No reviewer-rejection vector worth more attention than fabricated references.

---

## Methodology

Search bias was: prefer **EMSE / IEEE TSE / ACM TOSEM / ICSE / FSE / ASE / MSR / ESEM / ICSME** venues; arXiv accepted but flagged. Within each topic I selected (a) one canonical / heavily-cited reference and (b) one or two more recent (2022+) follow-ups. Where the original paper request mentioned a placeholder key (e.g., `li2024llm_bugs`), the goal is to find a replacement that stylistically fits the exact citation context (line-numbered in `00-paper-baseline.md`).

---

## Topic 1 — DDD adoption challenges (Section 2.1, line 146)

### 1.1 Mazlami et al. 2017 (already canonical) 🟢

- **Title:** Extraction of Microservices from Monolithic Software Architectures
- **Authors:** Mazlami, G.; Cito, J.; Leitner, P.
- **Venue/Year/DOI:** 2017 IEEE International Conference on Web Services (ICWS), pp. 524–531. DOI 10.1109/ICWS.2017.61.
- **Why it fits:** Section 2.1 — third-party empirical evidence that decomposition into bounded-context-aligned microservices is non-trivial; their formal extraction algorithms set up the contrast that DDD-Enforcer addresses *enforcement* of an already-decomposed system.
- **BibTeX:**
  ```bibtex
  @inproceedings{mazlami2017extraction,
    author={Mazlami, Genc and Cito, J{\"u}rgen and Leitner, Philipp},
    title={Extraction of Microservices from Monolithic Software Architectures},
    booktitle={2017 IEEE International Conference on Web Services (ICWS)},
    year={2017},
    pages={524--531},
    doi={10.1109/ICWS.2017.61}
  }
  ```

### 1.2 Kapferer & Zimmermann 2020 — Context Mapper 🟢

- **Title:** Domain-Driven Service Design — Context Modeling, Model Refactoring and Contract Generation
- **Authors:** Kapferer, S.; Zimmermann, O.
- **Venue/Year/DOI:** Symposium and Summer School on Service-Oriented Computing (SummerSoC) 2020. (Communications in Computer and Information Science, Springer.)
- **Why it fits:** Section 2.1 — companion citation to Mazlami; provides tooling around DDD context maps. Establishes that automated *modelling* tools exist for DDD but not enforcement.
- **BibTeX:**
  ```bibtex
  @inproceedings{kapferer2020dddservice,
    author={Kapferer, Stefan and Zimmermann, Olaf},
    title={Domain-Driven Service Design: Context Modeling, Model Refactoring and Contract Generation},
    booktitle={Symposium and Summer School on Service-Oriented Computing (SummerSoC)},
    year={2020},
    publisher={Springer}
  }
  ```

### 1.3 Bogner, Fritzsch et al. 2021 — Microservice maturity / DDD migration 🟡

- **Title:** Microservices in Industry: Insights into Technologies, Characteristics, and Software Quality
- **Authors:** Bogner, J.; Fritzsch, J.; Wagner, S.; Zimmermann, A.
- **Venue/Year/DOI:** ICSA 2019 / extended version JSEP 2021. *Verify the JSEP extended-version DOI.*
- **Why it fits:** Section 2.1 — surveys industry challenges with microservice maturity, several of which are DDD-rooted (boundary drift, vocabulary fragmentation).
- **BibTeX:**
  ```bibtex
  @article{bogner2021microservices_industry,
    author={Bogner, Justus and Fritzsch, Jonas and Wagner, Stefan and Zimmermann, Alfred},
    title={Microservices in Industry: Insights into Technologies, Characteristics, and Software Quality},
    journal={Journal of Software: Evolution and Process},
    year={2021},
    note={VERIFY DOI/volume}
  }
  ```

---

## Topic 2 — Architecture conformance checking (Section 2.2, line 160)

### 2.1 Knodel & Popescu 2007 — Reflexion-model survey 🟢

- **Title:** A Comparison of Static Architecture Compliance Checking Approaches
- **Authors:** Knodel, J.; Popescu, D.
- **Venue/Year/DOI:** WICSA 2007. DOI 10.1109/WICSA.2007.1.
- **Why it fits:** Section 2.2 — taxonomic baseline. ArchUnit and our work are concrete instances of "rule-based static checkers"; Knodel/Popescu situate the family.
- **BibTeX:**
  ```bibtex
  @inproceedings{knodel2007compliance,
    author={Knodel, Jens and Popescu, Daniel},
    title={A Comparison of Static Architecture Compliance Checking Approaches},
    booktitle={Working IEEE/IFIP Conference on Software Architecture (WICSA)},
    year={2007},
    doi={10.1109/WICSA.2007.1}
  }
  ```

### 2.2 Passos et al. 2010 — Reflexion + dependency rules 🟢

- **Title:** Static Architecture-Conformance Checking: An Illustrative Overview
- **Authors:** Passos, L.; Terra, R.; Valente, M. T.; Diniz, R.; Mendonça, N.
- **Venue/Year/DOI:** IEEE Software 2010, 27(5), 82–89. DOI 10.1109/MS.2009.117.
- **Why it fits:** Section 2.2 — the closest single paper to "what ArchUnit does and what its limitations are". Used as the "structural conformance is solved; semantic isn't" pivot.
- **BibTeX:**
  ```bibtex
  @article{passos2010conformance,
    author={Passos, Leonardo and Terra, Ricardo and Valente, Marco Tulio and Diniz, Renato and Mendon{\c{c}}a, Nabor},
    title={Static Architecture-Conformance Checking: An Illustrative Overview},
    journal={IEEE Software},
    year={2010},
    volume={27},
    number={5},
    pages={82--89},
    doi={10.1109/MS.2009.117}
  }
  ```

### 2.3 Architectural erosion (Li et al. 2022) 🟡

- **Title:** Why and How Software Architectural Erosion Persists: A Multi-Vocal Literature Review
- **Authors:** Li, R.; Liang, P.; et al.
- **Venue/Year/DOI:** Journal of Systems and Software (JSS), 2022. *Verify exact volume + DOI.*
- **Why it fits:** Section 2.2 — supports the "domain-model degradation" narrative we use as motivation in §1.
- **BibTeX:**
  ```bibtex
  @article{li2022architectural_erosion,
    author={Li, Ruiyin and Liang, Peng and others},
    title={Why and How Software Architectural Erosion Persists: A Multi-Vocal Literature Review},
    journal={Journal of Systems and Software},
    year={2022},
    note={VERIFY: full author list, volume, DOI}
  }
  ```

---

## Topic 3 — Microservice anti-pattern / decomposition quality (Section 2.3, line 173)

### 3.1 Taibi et al. 2017 — Microservice anti-patterns 🟢

- **Title:** Microservices Anti-Patterns: A Taxonomy
- **Authors:** Taibi, D.; Lenarduzzi, V.; Pahl, C.
- **Venue/Year:** Microservices: Science and Engineering, Springer 2020 (compiled from earlier 2017–2018 venue work).
- **Why it fits:** Section 2.3 — the canonical anti-pattern catalogue cited by virtually every microservice-quality paper since 2017. Connects DDD violations to recognized anti-patterns.
- **BibTeX:**
  ```bibtex
  @incollection{taibi2020microservices_antipatterns,
    author={Taibi, Davide and Lenarduzzi, Valentina and Pahl, Claus},
    title={Microservices Anti-Patterns: A Taxonomy},
    booktitle={Microservices: Science and Engineering},
    publisher={Springer},
    year={2020}
  }
  ```

### 3.2 Carrasco, van Bladel, Demeyer 2018 — Microservice migration smells 🟡

- **Title:** Migrating towards microservices: migration and architecture smells
- **Authors:** Carrasco, A.; van Bladel, B.; Demeyer, S.
- **Venue/Year:** International Workshop on Refactoring (IWoR) 2018, ASE companion. *Verify proceedings.*
- **Why it fits:** Section 2.3 — empirical migration smell catalogue; complements Taibi.
- **BibTeX:**
  ```bibtex
  @inproceedings{carrasco2018migration_smells,
    author={Carrasco, Alvaro and van Bladel, Brent and Demeyer, Serge},
    title={Migrating Towards Microservices: Migration and Architecture Smells},
    booktitle={Proceedings of the 2nd International Workshop on Refactoring (IWoR)},
    year={2018},
    note={VERIFY proceedings/page}
  }
  ```

### 3.3 Soldani, Tamburri, Van Den Heuvel 2018 — Pains and gains of microservices 🟢

- **Title:** The Pains and Gains of Microservices: A Systematic Grey Literature Review
- **Authors:** Soldani, J.; Tamburri, D. A.; Van Den Heuvel, W.-J.
- **Venue/Year:** Journal of Systems and Software, 2018, vol. 146. DOI 10.1016/j.jss.2018.09.082.
- **Why it fits:** Section 2.3 — backs the claim that microservice quality is itself a research-active topic; alternative-citation slot if Taibi is over-cited.
- **BibTeX:**
  ```bibtex
  @article{soldani2018pains,
    author={Soldani, Jacopo and Tamburri, Damian Andrew and Van Den Heuvel, Willem-Jan},
    title={The Pains and Gains of Microservices: A Systematic Grey Literature Review},
    journal={Journal of Systems and Software},
    year={2018},
    volume={146},
    pages={215--232},
    doi={10.1016/j.jss.2018.09.082}
  }
  ```

---

## Topic 4 — LLMs for software architecture (Section 2.4, line 182)

### 4.1 Ozkaya 2023 — IEEE Software keynote on LLMs in architecture 🟡

- **Title:** Application of Large Language Models to Software Engineering Tasks: Opportunities, Risks, and Implications
- **Authors:** Ozkaya, I.
- **Venue/Year:** IEEE Software, 2023, 40(3), 4–8. *Verify title — early Ozkaya editorials on LLMs+SE are slightly variant.*
- **Why it fits:** Section 2.4 — high-visibility editorial that frames LLM-for-architecture as an emerging research line; useful in introduction.
- **BibTeX:**
  ```bibtex
  @article{ozkaya2023llm_se,
    author={Ozkaya, Ipek},
    title={Application of Large Language Models to Software Engineering Tasks: Opportunities, Risks, and Implications},
    journal={IEEE Software},
    year={2023},
    volume={40},
    number={3},
    note={VERIFY exact title and pagination}
  }
  ```

### 4.2 Cito et al. 2024 — LLM-driven program comprehension 🟡

- **Title:** Towards LLM-Assisted Architecture Recovery (or similar title — verify)
- **Authors:** Cito, J.; et al.
- **Venue/Year:** ICSE / FSE 2024. **🔴 Verify exact paper.** Cito has multiple LLM-architecture papers.
- **Why it fits:** Section 2.4 — closest-domain reference for LLM-driven architecture analysis. If exact paper can't be confirmed, swap for the next entry below.
- **BibTeX:**
  ```bibtex
  @inproceedings{cito2024llm_architecture,
    author={Cito, J{\"u}rgen and others},
    title={(VERIFY: LLM Architecture Recovery / Architecture Smell Detection)},
    booktitle={(VERIFY: ICSE/FSE 2024)},
    year={2024},
    note={RISKY -- author has multiple titles in this space}
  }
  ```

### 4.3 Hou et al. 2024 — already in paper as `hou2024llm_se_survey` 🟢

Confirmed in current bibliography (line 956). No action needed; already used in §2.4.

### 4.4 Fan et al. 2023 — already in paper as `fan2023llm_se` 🟢

Confirmed in current bibliography (line 959). No action.

---

## Topic 5 — LLM-as-a-Judge in SE (Section 2.4, line 192; already cited)

### 5.1 Already in paper 🟢

The current bibliography has `llmjudgeSE2025`, `llmjudgeSurvey2024`, `llmjudgesComprehensive2024` (lines 991–998). These are sufficient for §2.4 context. **Recommendation:** Verify the arXiv IDs (2502.06193, 2411.15594, 2412.05579) actually resolve and the papers are still at those addresses.

### 5.2 Potential addition — Zheng et al. 2023 NeurIPS 🟢

- **Title:** Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
- **Authors:** Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.; Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E.; Zhang, H.; Gonzalez, J. E.; Stoica, I.
- **Venue/Year:** NeurIPS 2023 Datasets and Benchmarks Track. arXiv:2306.05685.
- **Why it fits:** Section 2.4 / §4.5 — the foundational LLM-as-judge paper outside SE; supports our methodological choice.
- **BibTeX:**
  ```bibtex
  @inproceedings{zheng2023judging,
    author={Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and Li, Zhuohan and Li, Dacheng and Xing, Eric P. and Zhang, Hao and Gonzalez, Joseph E. and Stoica, Ion},
    title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2023},
    eprint={2306.05685},
    archivePrefix={arXiv}
  }
  ```

---

## Topic 6 — Multi-agent LLM systems for SE (Section 2.6, line 215)

### 6.1 Bairi et al. 2024 — CodePlan 🟢

- **Title:** CodePlan: Repository-Level Coding using LLMs and Planning
- **Authors:** Bairi, R.; Sonwane, A.; Kanade, A.; Iyer, V.; Parthasarathy, S.; Rajamani, S.; Ashok, B.; Shet, S.
- **Venue/Year:** FSE 2024 / ICSE 2024 (one of the two; verify). arXiv:2309.12499.
- **Why it fits:** Section 2.6 — strongest single example of multi-agent LLM planning applied to a *repository-level* SE task; directly justifies our choice of multi-agent decomposition for rulebook extraction.
- **BibTeX:**
  ```bibtex
  @inproceedings{bairi2024codeplan,
    author={Bairi, Ramakrishna and Sonwane, Atharv and Kanade, Aditya and Iyer, Vageesh and Parthasarathy, Suresh and Rajamani, Sriram and Ashok, B. and Shet, Shashank},
    title={CodePlan: Repository-Level Coding using LLMs and Planning},
    booktitle={Proceedings of the ACM on Software Engineering (FSE 2024)},
    year={2024},
    eprint={2309.12499},
    archivePrefix={arXiv},
    note={VERIFY: FSE vs ICSE proceedings}
  }
  ```

### 6.2 Hong et al. 2024 — MetaGPT 🟢

- **Title:** MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework
- **Authors:** Hong, S.; Zheng, X.; Chen, J.; Cheng, Y.; Wang, J.; Zhang, C.; Wang, Z.; Yau, S. K. S.; Lin, Z.; Zhou, L.; Ran, C.; Xiao, L.; Wu, C.
- **Venue/Year:** ICLR 2024 (Oral). arXiv:2308.00352.
- **Why it fits:** Section 2.6 — most-cited general-purpose multi-agent LLM framework; supports the architectural choice.
- **BibTeX:**
  ```bibtex
  @inproceedings{hong2024metagpt,
    author={Hong, Sirui and Zheng, Xiawu and Chen, Jonathan and Cheng, Yuheng and Wang, Jinlin and Zhang, Ceyao and Wang, Zili and Yau, Steven Ka Shing and Lin, Zijuan and Zhou, Liyang and Ran, Chenyu and Xiao, Lingfeng and Wu, Chenglin},
    title={MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2024},
    eprint={2308.00352},
    archivePrefix={arXiv}
  }
  ```

### 6.3 Qian et al. 2024 — ChatDev 🟢

- **Title:** ChatDev: Communicative Agents for Software Development
- **Authors:** Qian, C.; Liu, W.; Liu, H.; Chen, N.; Dang, Y.; Li, J.; Yang, C.; Chen, W.; Su, Y.; Cong, X.; Xu, J.; Li, D.; Liu, Z.; Sun, M.
- **Venue/Year:** ACL 2024. arXiv:2307.07924.
- **Why it fits:** Section 2.6 — third pillar of the multi-agent SE landscape; Bairi+Hong+Qian collectively cover the space.
- **BibTeX:**
  ```bibtex
  @inproceedings{qian2024chatdev,
    author={Qian, Chen and Liu, Wei and Liu, Hongzhang and Chen, Nuo and Dang, Yufan and Li, Jiahao and Yang, Cheng and Chen, Weize and Su, Yusheng and Cong, Xin and Xu, Juyuan and Li, Dahai and Liu, Zhiyuan and Sun, Maosong},
    title={ChatDev: Communicative Agents for Software Development},
    booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
    year={2024},
    eprint={2307.07924},
    archivePrefix={arXiv}
  }
  ```

### 6.4 AutoGen — already in paper as `wu2023autogen` 🟢

Confirmed (line 977). Sufficient.

---

## Topic 7 — LLM bug detection (replacement for `li2024llm_bugs`)

### 7.1 Replacement candidate — Sun et al. 2024 / Yin et al. 2024 🟡

- **Title:** A Survey on Large Language Models for Software Engineering (or specifically: bug detection sub-survey within Hou et al. 2024 already cited)
- **Recommendation:** Since `hou2024llm_se_survey` (line 956) already covers this comprehensively, the simplest fix is to replace `\cite{li2024llm_bugs}` with `\cite{hou2024llm_se_survey}` for the "bug detection" citation in line 178 — both citations point to LLM-SE surveys, so the swap loses nothing in §2.4.
- **Alternative:** Cite a specific empirical paper:
  - **Title:** Large Language Models for Test-Free Fault Localization
  - **Authors:** Yang, A. Z. H.; Le Goues, C.; Martins, R.; Hellendoorn, V. J.
  - **Venue:** ICSE 2024. arXiv:2310.01726.
  - **Why it fits:** Concrete empirical use of LLMs for bug-detection-adjacent task. 🟢
  - **BibTeX:**
    ```bibtex
    @inproceedings{yang2024fault_localization,
      author={Yang, Aidan Z. H. and Le Goues, Claire and Martins, Ruben and Hellendoorn, Vincent J.},
      title={Large Language Models for Test-Free Fault Localization},
      booktitle={Proceedings of the 46th IEEE/ACM International Conference on Software Engineering (ICSE)},
      year={2024},
      eprint={2310.01726},
      archivePrefix={arXiv}
    }
    ```

---

## Topic 8 — LLM program comprehension (replacement for `nam2024llm_comprehension`)

### 8.1 Nam et al. 2024 — Using LLMs for code understanding 🟡

- **Title:** Using an LLM to Help with Code Understanding
- **Authors:** Nam, D.; Macvean, A.; Hellendoorn, V.; Vasilescu, B.; Myers, B.
- **Venue/Year:** ICSE 2024. **Verify the precise title — Nam has multiple titles in this orbit.**
- **Why it fits:** Section 2.4 — directly slots into the placeholder citation for "program comprehension". A real paper.
- **BibTeX:**
  ```bibtex
  @inproceedings{nam2024llm_comprehension,
    author={Nam, Daye and Macvean, Andrew and Hellendoorn, Vincent and Vasilescu, Bogdan and Myers, Brad},
    title={Using an LLM to Help with Code Understanding},
    booktitle={Proceedings of the 46th IEEE/ACM International Conference on Software Engineering (ICSE)},
    year={2024},
    note={VERIFY exact title (Nam has multiple LLM-comprehension papers)}
  }
  ```

---

## Topic 9 — DDD + LLM industry writeups (`dddllmindustry2024`)

### 9.1 Sebastian 2024 — Enhancing DDD with Generative AI 🔴

- **Title:** Enhancing Domain-Driven Design with Generative AI (or similar; reported on Medium / company blogs in 2024)
- **Author:** Sebastian (single-name pseudonym on Medium; verify before citing)
- **Venue/Year:** Industry blog post, 2024.
- **Why it fits:** §2.5 — a real, industry-perspective LLM-DDD writeup. Listed in the original prompt; flagged as RISKY because Medium-only / non-peer-reviewed citations are weaker. Recommend supplementing rather than relying.
- **BibTeX:**
  ```bibtex
  @misc{sebastian2024ddd_generativeai,
    author={Sebastian, B.},
    title={Enhancing Domain-Driven Design with Generative AI},
    howpublished={Medium},
    year={2024},
    note={VERIFY author name and URL; non-peer-reviewed}
  }
  ```

### 9.2 Mülder 2024 — LLM-constrained DSL generation for DDD 🔴

- **Author:** Mülder (verify)
- **Year:** 2024
- **Venue:** likely a conference talk / industry writeup
- **Status:** Reported in the planning prompt; cannot verify from training data.
- **Recommendation:** Verify or replace.

### 9.3 Evans 2024 InfoQ — already in paper as `evans2024infoq` 🟢

Confirmed (line 985). Best choice for a 1-citation industry slot. Could fold `dddllmindustry2024` into a single "industry-perspective evidence" sentence backed by Evans+Sebastian (if confirmed).

---

## Topic 10 — `automatingddd2024` arXiv ID forensics

**Statement:** The current bibliography (paper.tex line 982) cites `arXiv:2603.26244` for a 2024 "Automating Domain-Driven Design: Experience with a Prompting Framework" paper.

**Diagnosis (high confidence):**
- arXiv IDs follow `YYMM.NNNNN`. `2603` parses as **March 2026**, contradicting the claimed 2024 date.
- Even read as a 2026 ID, suffix `26244` is at the high end of monthly volume.
- The URL `https://arxiv.org/abs/2603.26244` will not resolve.

**Recommendations (in order of preference):**

1. **🟢 Most likely real paper to substitute (verify):** *Anwar 2024* (or similar) "Automating Domain-Driven Design with Large Language Models" or "Domain-Driven Design Patterns with LLMs". **Action:** Search arXiv 2024 for "domain-driven design" + "LLM" or "prompting". The original paper the author had in mind likely has an arXiv ID in the `2401.NNNNN` to `2412.NNNNN` range.

2. **🟢 If no real paper can be located:** Delete the citation entirely. Adjust prose at:
   - L91: "A small but growing body of work has begun applying LLMs to DDD itself~\cite{automatingddd2024, evans2024infoq}" → drop `automatingddd2024`, retain `evans2024infoq` and `dddllmindustry2024`.
   - L198: "The prompting-framework proposal of~\cite{automatingddd2024} decomposes DDD into five sequential steps…" → rewrite as "Industry experimentation with LLM-assisted DDD~\cite{evans2024infoq,dddllmindustry2024} typically decomposes DDD into…" — losing specificity but keeping the 5-step framing.
   - L224: "Unlike prior LLM-assisted DDD work~\cite{automatingddd2024,evans2024infoq}…" → drop `automatingddd2024`.
   - L789: same as L224.

3. **🔴 Do NOT submit with the impossible arXiv ID.** This is a desk-reject magnet for any reviewer who clicks the URL.

**Routes to:** WP-10 — explicitly track `automatingddd2024` resolution as a sub-task with sign-off from Baran (since he originally added the citation, he may have provenance for the paper).

---

## Open questions / candidates that did not make the cut

- **DDD adoption challenges (Topic 1):** I prioritized Mazlami / Kapferer / Bogner. If reviewer wants more, candidates: Vural et al. 2017 (microservice migration), Pahl & Jamshidi 2016 (microservices systematic review).
- **LLM-as-Judge in SE specifically published in EMSE:** I could not name a specific EMSE-published LLM-as-judge paper from memory. If such a paper exists (2024–2026), it would be ideal as an EMSE-targeted citation. Literature subagent should retry once web tools are restored.
- **Practitioner empirical surveys on DDD (would strengthen WP-09 framing):** Mendonça et al. or similar — verify.

---

## Submission checklist for WP-10

- [ ] Verify each 🟢 entry against publisher / arXiv (DOI must resolve)
- [ ] Verify each 🟡 entry; downgrade to 🔴 or drop if unverifiable
- [ ] Remove or replace each 🔴 entry; do not ship under any circumstances
- [ ] Replace `\bibitem{}` inline format with `refs.bib` BibTeX file (Springer style: `\bibliographystyle{spbasic}` + `\bibliography{refs}`)
- [ ] Resolve `automatingddd2024` arXiv ID — keep with a real ID, replace, or delete
- [ ] Fill `dincoguz2025ddd_conference` — Baran has provenance (UBMK 2025 details)
- [ ] Resolve `llmcodequality2026` author list (DOI 10.1007/s10664-026-10858-8 — verify EMSE has accepted/published)
- [ ] Final pre-submission grep: `grep -n placeholder paper.tex` returns zero hits
