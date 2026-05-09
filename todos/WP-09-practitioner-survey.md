# WP-09: Practitioner Survey — ❌ DROPPED FROM ACTIVE SCOPE

> **STATUS — 2026-05-08**: This WP is **DROPPED** from the active 14-week roadmap (D6 decision: RQ5 silindi → WP-09'un dayanağı olan "developer study" variantı kapsamdan çıktı).
>
> - **Reason**: WP-09 was tied to RQ5 (Developer Study variant). With RQ5 dropped (`MASTER_PLAN.md` D6), the IRB+recruitment cost is no longer justified for a 4-RQ paper.
> - **Reviewer impact**: None. RQ5 is silently removed; reviewers won't ask "where is the survey?"
> - **Hoca communication**: Murat Hoca courtesy bilgilendirilir (`HOCA_GUNDEM.md` Konu 2).
> - **Future**: If reviewers request major revision with practitioner study angle, this WP can be reactivated.
>
> **The original content below is preserved for historical reference only. Do NOT execute as part of EMSE submission.**

---

# WP-09: Practitioner Survey (Industry Relevance — Optional but Recommended) [ARCHIVED]

**Owner:** Baran (recruit + IRB + analysis)
**Depends-on:** [WP-00] (concurrent with infra; IRB delay tolerated)
**Effort:** L (IRB ~3 weeks + recruitment + survey + analysis)
**Status:** TODO
**Addresses instructor feedback:** [B.6 reviewer threat: industry relevance — EMSE-special]
**Decision gate:** May merge with WP-18 if RQ5 = (D) Developer Study.

## Goal

Recruit 12–20 microservice developers (LinkedIn + TEDU alumni + meetup networks). Show each a curated set of framework-detected violations from RQ3/RQ4 runs. Ask: "useful / wrong / neither?" — collect quantitative agreement + qualitative themes (which violation types are most actionable, where the framework helps vs. hinders, what they'd want from a v2). Output: §"Practitioner Perspective" subsection in §8 (RQ3) or §9 (Discussion); industry-relevance signal that EMSE editors prize.

## Acceptance criteria

- [ ] TEDU IRB approval secured (start application week 1).
- [ ] ≥12 participants recruited; demographic spread documented (years of experience, microservice exposure, DDD familiarity).
- [ ] Survey instrument: pre-test (DDD background), violation-rating task (15–20 violation reports per participant), post-test (open-ended impressions).
- [ ] Each violation report rated on 4-point Likert: "Definitely useful / Probably useful / Probably wrong / Definitely wrong / Cannot tell".
- [ ] Cohen's κ between participants computed for inter-rater reliability on shared violations.
- [ ] Qualitative analysis (open-coding by Baran on free-text responses) identifying themes: most-actionable violation types, friction points, must-haves for v2.
- [ ] §"Practitioner Perspective" subsection written: 0.5–1 page, summary numbers + 2–3 representative quotes (anonymized).
- [ ] Replication package includes anonymized survey responses (`replication_package/practitioner_survey/`).

## Implementation steps

1. **Week 1:** Draft TEDU IRB application: study purpose, recruitment text, instrument, consent form, data handling. Submit immediately.
2. **Week 2:** Draft recruitment text + post on LinkedIn + email TEDU SE alumni list + post in 2 microservice/DDD meetups.
3. **Week 3:** IRB feedback + revisions; recruit in parallel (target 20 to net 12).
4. **Week 4:** IRB approval (typical TEDU timeline). Send survey to confirmed participants.
5. **Week 5–6:** Survey responses arrive; reminders sent.
6. **Week 7:** Close survey; compute κ; open-code free-text.
7. **Week 8:** Write §"Practitioner Perspective" subsection.
8. Decide: stays as appendix subsection OR becomes RQ5=(D) (then merge with WP-18).

## Outputs (file paths)

- `survey/irb_approval.pdf`
- `survey/instrument.pdf` (survey text)
- `survey/recruitment_text.md`
- `survey/responses_anonymized.csv`
- `survey/themes.md` (open-coding output)
- `paper.tex` §"Practitioner Perspective" subsection
- `replication_package/practitioner_survey/`

## Risks & mitigations

- **Risk:** IRB rejection or extended timeline. **Mitigation:** Submit week 1, plan for 4-week IRB cycle; if rejected, address feedback within 1 week. Worst case: drop WP-09 (R3 in `01-risks.md`).
- **Risk:** Recruitment shortfall (target 12, achieve 6). **Mitigation:** Reframe as "case study" with n ≥ 5; qualitative themes still publishable. Frame in §4.x as "small-sample qualitative probe, not representative survey".
- **Risk:** Participants unfamiliar with DDD distort responses. **Mitigation:** Pre-test screens for DDD familiarity; per-participant κ computed; low-DDD-familiarity responses analyzed separately.
- **Risk:** Survey takes too long per participant (drop-out). **Mitigation:** Limit to 15 violations max, target 20 minutes total; pilot-test with 2 participants first.
