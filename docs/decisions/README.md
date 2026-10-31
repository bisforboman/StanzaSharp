# Decisions

The record of what was decided, by whom and when, with the measurements behind it. CLAUDE.md keeps only the rules that
follow from these; it links each file below. Until 2026-10-31 all of this lived in CLAUDE.md (git history has it).

- [scope.md](scope.md): languages, processors, packages, per-processor models, transformers, coref.
- [status.md](status.md): each processor's port, what it reproduces, input modes, options, the performance rounds' outcomes.
- [validation.md](validation.md): golden data, what `tools/make_golden.py` writes, test conventions.
- [packaging-and-releases.md](packaging-and-releases.md): packages, public API, downloads, versions, releases.
- [managed-backend.md](managed-backend.md): the managed backend's decisions, the seam per processor, AOT.
- Outside this folder: [../ci.md](../ci.md) (CI), [../checkpoint-findings.md](../checkpoint-findings.md),
  [../model-files.md](../model-files.md), and docs/performance/ (one file per measurement round).

A new decision goes into the file for its area, dated and attributed ("owner's decision, YYYY-MM-DD"), next to what it
changes; a decision that replaces an earlier one says so there rather than deleting it. If it changes a rule, update
CLAUDE.md too.
