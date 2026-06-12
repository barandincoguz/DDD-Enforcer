# AGENTS.md

## Purpose

Build this project in a modular, simple, and maintainable way.

Prefer:

- clarity over cleverness,
- small safe changes over broad rewrites,
- stable entrypoints over fragile convenience,
- and explicit structure over hidden complexity.

The goal is not just to make the code work, but to keep it understandable, extensible, and easy to debug.

---

## Core Rules

- Always think before coding.
- Understand the workflow, architecture, and surrounding context before making changes.
- Solve the problem with the smallest correct change.
- Keep solutions modular, simple, and easy to reason about.
- Avoid overengineering.
- Do not make unrelated changes.
- Do not expand scope without a strong reason.
- Prefer maintainability over speed of implementation.

---

## Modularity

- Build modular first. Keep responsibilities narrow and clear; prefer focused modules over multi-purpose ones.
- **File size is a guideline, not a wall.** Counted as effective lines (excluding comments + blank lines):
  - **~500 lines** is the sweet spot — readable in one pass, fits in context comfortably.
  - **~800 lines** is a review trigger — during refactor or code review, ask "is this still one responsibility?" If yes, leave it. If it has drifted into mixed concerns, split.
  - **~1200+ lines** is a pressure point — only justified when splitting would create artificial coupling. Otherwise split.
- Splitting a cohesive file just to hit a number is worse than leaving it intact. Split by responsibility, not by line count.
- Files that change together should live together. Use composition and separation of concerns; do not contort a clean design to satisfy a size target.
- If a file starts accumulating *multiple unrelated responsibilities*, split it regardless of size.

---

## Simplicity First

- Prefer the simplest solution that fully solves the problem.
- Do not introduce abstractions unless they clearly improve the design.
- Avoid speculative generalization.
- Avoid unnecessary indirection, wrappers, layers, and helpers.
- Do not design for imaginary future requirements.
- However, if a future change is already clearly expected, isolate the change-prone logic now so entrypoints remain stable.

---

## Design Principles

- Keep entrypoints stable.
- Isolate changing or complex logic into smaller internal modules.
- Prefer simple and explicit flows over clever or deeply abstract designs.
- Keep boundaries between layers clear.
- Avoid hidden side effects.
- Avoid tightly coupling unrelated concerns.
- Write code that is easy for another engineer to understand quickly.
- Do not write code that you already know will need redesign soon.

---

## Change Discipline

Before coding:

1. Understand the real problem.
2. Understand the relevant workflow, architecture, and context.
3. Identify the smallest safe implementation.
4. Place logic in the correct module.
5. Check whether the change preserves modularity and simplicity.

When coding:

- Keep changes small and focused.
- Preserve clear boundaries.
- Avoid scope creep.
- Prefer readability over cleverness.
- Do not refactor unrelated parts of the codebase.
- Do not rename symbols, files, or modules unless necessary.
- Do not mix feature work with broad cleanup unless explicitly requested.

After coding:

- Check whether the solution is still the simplest viable one.
- Check whether any file became too large or too broad in responsibility.
- Check whether the change introduced hidden coupling or unnecessary abstraction.

---

## Error Handling

- During development, do **not** add default fallbacks to hide failures.
- If something fails, let it fail clearly so it can be fixed properly.
- Prefer explicit failure over silent degradation.
- Never leave empty `try/catch` blocks.
- Never swallow exceptions silently.
- If an exception is caught, handle it intentionally:
  - add useful context,
  - convert it into a clear failure,
  - or rethrow it.
- Do not hide bugs behind permissive defaults, silent retries, or vague recovery paths unless explicitly required.

---

## Dependencies

- Do not reinvent the wheel.
- Prefer solid open-source, self-hostable libraries when appropriate.
- Use mature, understandable, and well-maintained libraries.
- When library choice matters, help the user evaluate options.
- Do not add unnecessary dependencies.
- Every dependency should solve a real problem and justify its cost.
- Prefer fewer strong dependencies over many weak ones.

---

## UI Guidance

- Design UI for the end user, not for the schema.
- Do not expose internal structure directly just because it is convenient.
- Favor usability, clarity, and task flow.
- Backend structure should support the UI, not dictate it.
- Prefer interfaces that feel simple and intentional to the user.

---

## Workflow Awareness

- Always understand how the affected code fits into the broader workflow.
- Do not apply local fixes that create global inconsistency.
- Respect the architecture already present in the repository unless change is necessary.
- Before modifying a module, understand its inputs, outputs, and dependencies.
- Preserve consistency with the project’s existing patterns when those patterns are sound.

---

## Implementation Style

Preferred style:

- small functions,
- focused modules,
- explicit control flow,
- clear naming,
- low cognitive load,
- stable entrypoints,
- and debuggable behavior.

Avoid:

- giant files,
- giant functions,
- hidden behavior,
- premature abstraction,
- unnecessary indirection,
- schema-driven UI decisions,
- and broad speculative refactors.

---

## Final Reminder

Modularity is good.  
Simplicity is good.  
Clarity is good.  
Overengineering is bad.

Think first.  
Understand the workflow.  
Then implement the smallest correct solution.


<claude-mem-context>
# Memory Context

# [DDD-Enforcer] recent context, 2026-05-26 1:10pm GMT+3

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (19,745t read) | 269,549t work | 93% savings

### May 25, 2026
S1025 Architectural clarification of C (Holistic Critic) — finalized report-only design positioned as 7th pipeline stage with detailed control flow and scope boundaries (May 25 at 1:57 PM)
S1026 High-level purpose clarification for C (Holistic Critic) — simplified explanation of role, scope, and why design avoids automatic mutation (May 25 at 1:58 PM)
S1027 Decision between three topologies for Critic (C) component feedback loop: regenerate (A), in-place revise (B), or single agentic agent (C) with typed tools (May 25 at 1:59 PM)
S1028 Design and validate "Topology A — Active Critique Loop" architecture for LLM-augmented DDD pipeline, replacing report-only spec with executable loop supporting Critic feedback, routing, and iterative refinement (May 25 at 2:07 PM)
S1029 Holistic Critic (C) component design spec written and committed; decision gate resolved (product quality focus); awaiting next phase direction (May 25 at 2:12 PM)
S1030 Ship Holistic Critic (C) component via SDD process with three-stage review, merge to main, update memory, and identify next work (Context-Mapper A and UI/UX follow-up) (May 25 at 2:17 PM)
S1031 Flip DDD_CRITIC_LOOP default from opt-in (OFF) to opt-out (ON) post-shipping; update tests and validate gate passage (May 25 at 8:43 PM)
S1032 Close out Critic (C) shipping and default-flip; establish Context-Mapper (A) handoff with decision gates for next session (May 25 at 8:46 PM)
S1033 Architect Phase context-mapper design finalization: resolve three decision gates (schema, run-mode, taxonomy) and present complete design for Codex adversarial review (May 25 at 8:49 PM)
4940 9:18p 🔵 Adversarial Design Review: Context-Mapper Integration Blockers
4941 9:19p 🔵 Code Inspection Confirms Codex Blockers: AST Overwrites, D11 Missing, Call Graph Constraints
4942 " 🔵 Main.py Sequential Ordering Confirms AST Overwrite Risk; D11 Implementation Exists But Unused
4943 9:22p ✅ Context-Mapper Design Spec Created: All 10 Adversarial Review Fixes Integrated
4944 9:23p ✅ Context-Mapper Design Spec Finalized and Committed (0d2f542)
4945 9:28p 🟣 Holistic Critic Loop Shipped with Default ON
4946 " ✅ Decision Gate and Context Mapping Handoff Document Created
4947 " ✅ Multi-Project Memory Configuration Established
4948 " 🔵 Pipeline Test Status and Architecture State
4949 " ⚖️ Context-Mapper (A) Phase Implementation Plan Created
4950 9:29p ✅ Task 11 Updated to Maintain Suite Green During 4-tuple Partition Refactor
4951 " ✅ Context-Mapper Phase A Implementation Plan Committed to Repository
4952 9:30p ✅ Context-Mapper Implementation Tasks Created in Task Tracking System
4953 " ✅ 14-Task Context-Mapper Implementation Decomposed Into Task Tracking System
4954 " ✅ Context-Mapper Implementation Ready for Execution: All 14 Tasks Decomposed and Task 1 Started
4955 9:31p 🟣 Task 1 TDD Implementation Begins: Context Mapper Schema Tests Created
4956 " ✅ Task 1: Schema File Prepared with Required Imports
4957 " 🟣 Task 1 Implementation Complete: ContextRelationship + ContextMap Schema Classes Added
4958 " 🟣 Task 1 Complete: DomainModel.context_map Field Added, Schema Implementation Finalized
4959 9:32p 🟣 Task 1 Tests Pass: Schema Implementation Validated
4960 " 🟣 Task 1 Gate Verification Pass: Pyright Type Check 0 Errors
4961 9:33p 🟣 Task 1 Regression Check Pass: Full Unit Suite Green (769 Tests)
4962 " ✅ Task 1 Implementation Staged for Commit (Clean)
4963 " ✅ Task 1 Committed: ContextRelationship + ContextMap Schema Complete
### May 26, 2026
4964 3:19a 🔵 4-way partition implemented in loop.py; relationship test file missing
4965 " 🔵 Relationship and advisory partitions exist but unused in critique loop routing
4966 3:20a 🔵 Relationship findings extracted but no routing adapter implemented
4967 " 🟣 Relationship-only cycle routing implemented via context_mapper
4968 " 🔴 Relationship signature canonicalization implemented
4969 " 🟣 Relationship-only routing implemented in critique loop
4970 " 🟣 Relationship routing tests passing
4971 3:21a 🔵 Full test suite passes with relationship routing implementation
4972 " 🔵 Type safety and critic test suite validation
4973 " ✅ Relationship routing feature committed
4974 3:22a 🟣 Task 12 relationship-only loop routing completed
4975 " 🔵 Complete relationship routing feature integration verified
4976 3:23a 🔵 Final comprehensive validation: all 7 checks passed
4977 3:24a 🔵 Canonicalizer purity and pair filtering verified
4978 " 🔵 Structural and relationship findings coexist; remap is additive not exclusive
4979 3:25a 🔵 Best-cycle tracking and deep-copy purity verified across relationship-only cycles
4980 " 🔵 Signature deduplication semantics clarified
4981 " 🔵 Final integration test suite passes; all modules type-safe
4982 3:26a 🔵 Full test suite green; feature deployment ready
4983 3:27a ⚖️ Task 12 independently reviewed and approved for production
4984 3:28a 🟣 Context-mapper authority check added to import topology autofill
4985 " 🟣 Authoritative map blocks autofill; records drift for review
4986 3:29a 🔵 Task 13 validation: new tests passing, no regressions in full suite
4987 " ✅ Task 13 implementation committed
4988 " 🟣 Task 13 completed and independently verified
S1036 Memory checkpoint: Context-Mapper (A) feature delivery completion and project state transition to UI/UX phase (May 26 at 3:45 AM)
5004 1:10p ✅ Architecture documentation created for multi-agent DDD pipeline

Access 270k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>