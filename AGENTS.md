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

- Build modular first.
- No code file may exceed **400 lines (without comment lines and sections )**.
- If more code is needed, split it into more files, modules, or helper functions.
- Do not keep growing a file just to move faster.
- Keep responsibilities narrow and clear.
- Prefer small focused modules over large multi-purpose files.
- If a file starts accumulating multiple responsibilities, split it.
- Use composition and separation of concerns instead of monolithic designs.

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
