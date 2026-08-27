---

name: dead-code-cleanup
description: Remove unnecessary functions, wrappers, dead code, unused extras, and low-value indirection only from files changed in the most recent commit, while preserving behavior and public interfaces.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Skill: Dead Code and Wrapper Cleanup

## Purpose

Use this skill when asked to clean up code by removing:

* unused functions
* unused classes
* unused imports
* dead branches
* redundant wrapper functions
* unnecessary helper functions
* duplicate logic
* low-value abstraction
* stale comments
* unreachable code
* leftover debugging code
* unnecessary intermediate variables
* unnecessary re-export layers
* unused constants
* unused configuration fragments

The goal is to make the code simpler without changing behavior.

---

## Hard Scope Constraint

Only inspect and modify files that were changed in the **most recent commit**.

Do not edit files outside the last commit’s changed-file set.

The changed-file set must be determined with:

```bash
git diff --name-only HEAD~1..HEAD
```

or, if renamed files must be included:

```bash
git diff --name-status HEAD~1..HEAD
```

Only files listed by this command are in scope.

If a cleanup opportunity requires editing a file that was not changed in the last commit, do not edit it. Instead, mention it separately under “Out-of-scope cleanup opportunities.”

---

## Safety Rules

Do not remove code only because it looks unused.

Before removing anything, check whether it is referenced by:

* imports
* route registration
* dependency injection
* decorators
* reflection
* dynamic imports
* framework conventions
* configuration files
* tests
* CLI entrypoints
* public APIs
* exported modules
* package `__init__.py` files
* frontend routing
* Streamlit page discovery
* FastAPI / Flask route decorators
* Celery / scheduler registration
* plugin systems
* callbacks
* event handlers
* notebook or script usage
* external consumers

When uncertain, preserve the code and add it to “Needs human review.”

---

## Primary Objective

Simplify only the files changed in the last commit while preserving behavior.

Prefer small, safe cleanups over aggressive rewrites.

The final result should:

* reduce unnecessary code
* improve readability
* avoid behavior changes
* keep public interfaces stable
* keep tests passing
* avoid broad refactors
* avoid touching unrelated files

---

## Workflow

### 1. Identify the last-commit file set

Run:

```bash
git diff --name-only HEAD~1..HEAD
```

Save this list mentally as the only editable scope.

If the command returns no files, stop and report that there are no files changed in the last commit.

Ignore generated files, lock files, binary files, images, and vendored files unless explicitly requested.

---

### 2. Inspect only those files

For each changed source file:

* read the file
* understand its purpose
* identify functions, classes, constants, imports, and wrappers
* check whether each item is used inside the changed-file set
* search the repository for references before deleting anything

Use read-only search commands such as:

```bash
grep -R "function_or_class_name" .
```

or language-aware tools if available.

Do not rely only on text search when framework magic may be involved.

---

### 3. Classify cleanup candidates

Classify each candidate as one of:

| Category                 | Meaning                                    | Action                   |
| ------------------------ | ------------------------------------------ | ------------------------ |
| Safe removal             | Definitely unused or unreachable           | Remove                   |
| Safe simplification      | Wrapper or indirection with no value       | Inline or simplify       |
| Suspicious but uncertain | Might be used dynamically                  | Keep and report          |
| Public API risk          | Exported or externally callable            | Keep unless clearly safe |
| Out of scope             | Requires editing files outside last commit | Do not edit              |

---

## What Counts as Unnecessary

### Unused functions

A function may be removed when:

* it is not referenced anywhere
* it is not exported as public API
* it is not used by decorators, routes, callbacks, or config
* it is not part of an interface or abstract contract
* it is not used by tests or external entrypoints

---

### Redundant wrapper functions

A wrapper may be removed or inlined when it only does something like:

```python
def get_data():
    return load_data()
```

or:

```python
def render():
    return render_component()
```

and it adds no:

* validation
* logging
* error handling
* caching
* naming clarity
* type conversion
* abstraction boundary
* stable public API
* test seam
* framework hook
* domain meaning

Do not remove wrappers that clarify domain concepts or protect the rest of the code from implementation details.

---

### Dead branches

Remove unreachable branches such as:

* `if False`
* old feature-flag paths that can no longer run
* branches after `return`, `raise`, `continue`, or `break`
* impossible conditions confirmed by current types or constants

Do not remove feature-flagged code unless the flag is proven obsolete.

---

### Unused imports

Remove imports that are not used in the file.

Be careful with imports that exist for side effects, such as:

* plugin registration
* model registration
* route registration
* monkey-patching
* package initialization

If an import appears side-effect-only, keep it unless clearly safe.

---

### Duplicate helpers

If a helper duplicates logic inside the same changed-file set and can be safely consolidated without broad refactoring, simplify it.

Do not consolidate across files if that requires editing files outside the last commit.

---

### Low-value abstractions

Remove or inline abstractions that:

* are used only once
* obscure simple logic
* create unnecessary navigation
* do not represent a real domain concept
* do not isolate external systems
* do not improve testability

Do not remove abstractions that are part of a clean architectural boundary.

---

## Commands to Use

### Find files changed in the last commit

```bash
git diff --name-only HEAD~1..HEAD
```

### See exact changes from last commit

```bash
git diff HEAD~1..HEAD -- path/to/file
```

### Search for references

```bash
grep -R "symbol_name" .
```

### Python-specific checks

```bash
python -m compileall .
```

If available:

```bash
ruff check .
vulture .
pytest
```


### Git status check

Before finishing:

```bash
git status --short
git diff --stat
git diff
```

---

## Editing Rules

When editing:

* keep changes minimal
* do not reformat unrelated code
* do not rename files
* do not change behavior
* do not change public interfaces unless clearly safe
* do not modify tests unless explicitly asked
* do not edit files outside the last commit’s changed-file set
* do not perform broad architecture refactors
* do not remove comments that explain non-obvious business logic
* remove comments that only restate obvious code
* preserve typing, error handling, logging, and validation

---

## Validation

After cleanup, validate using the safest available commands.

Prefer project-specific commands from:

* `README.md`
* `pyproject.toml`
* `package.json`
* `Makefile`
* CI workflow files

If running full tests is too expensive, run targeted tests or static checks.

If no validation command is available, say so clearly.

---

## Required Final Response

The final response must include:

```markdown
# Dead Code Cleanup Summary

## Scope

Files changed in the most recent commit:

- `file1`
- `file2`

Only these files were edited.

## Removed

List removed functions, imports, wrappers, branches, constants, or comments.

## Simplified

List wrappers or abstractions that were simplified or inlined.

## Kept intentionally

List suspicious code that looked removable but was kept, with the reason.

## Out-of-scope cleanup opportunities

List cleanup opportunities found outside the last-commit file set, without editing them.

## Validation

List commands run and whether they passed.

## Risk notes

Mention any possible behavior risk or uncertainty.
```

---

## Important Judgment Rules

A wrapper is not unnecessary if it:

* gives domain meaning to a generic operation
* hides implementation details
* is used as a callback
* is used by a framework
* is part of public API
* stabilizes imports
* helps testing
* centralizes logging, validation, caching, or error handling

A function is not dead code if it is:

* referenced dynamically
* discovered by naming convention
* registered by decorator
* imported by external users
* part of a CLI
* part of a framework lifecycle
* used by tests
* part of a public module contract

When in doubt, do not delete. Report the uncertainty instead.

---

## Final Quality Bar

The cleanup is successful only if:

* all edits are limited to files changed in the most recent commit
* unnecessary code is removed safely
* behavior is preserved
* validation is attempted
* uncertain removals are not made
* the final summary clearly explains what changed and why
