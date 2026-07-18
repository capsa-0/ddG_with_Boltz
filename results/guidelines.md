# results/ — guidelines

How to structure a result folder so anyone (including a future Claude session or a
colleague) can pick it up without re-deriving context. Everything under `results/`
is **committed** (unlike `data/processed/`) — including this `guidelines.md` and
every `status.md` — so it syncs via git to the cluster and back.

## One folder per result

Name folders `NN_short_slug` in the order they were produced (`01_generalization`,
`02_stress_extrapolation`, …). Each folder is **self-contained**.

### Required in every result folder

| File | Purpose |
|---|---|
| **`README.md`** | What / Why / How + headline numbers + a **data & provenance table** (config path, source dataset, processed dir, feature table, benchmark output, code entry point). This is the first thing to read. |
| **`status.md`** | Living log of state and progress — see below. **Always present**, even for a planned/blocked experiment that has no results yet. |
| **`figures/`** | Numbered PNGs (`01_*.png`, …) + a `figures/README.md` index. **Required once the experiment has results** (a completed experiment always has at least one figure). |
| **`report.pdf`** | Paper-facing narrative write-up. **Required once the result is settled.** Prefer a committed `build_report.py` that regenerates it from the result tables + figures (see the report rule below). |

A **planned or still-running** experiment needs only `README.md` (may be a stub) and
`status.md`; `figures/` and `report.pdf` become required as soon as it has settled results.

### The `report.pdf` rule — paper-facing only

**Include only what would appear in a paper** — motivation, methods, results,
figures/tables, and interpretation. **Exclude** everything that is not part of the
scientific story: data provenance and file paths, pipeline/plumbing details, job IDs,
and any account of problems encountered or how the corpus was assembled (partial runs,
bug fixes, retries). Those belong in `status.md` (the debugging/work log), the README
**provenance table**, and `details.md` — never in `report.pdf`.

### Recommended

- **`details.md`** — methods/provenance appendix: exact hyperparameters, split
  definitions, per-number provenance behind the README's summary statements.
- **Result tables** — the raw `*.csv` / `*.json` the figures and headline numbers
  come from (e.g. `benchmark_summary.csv`, `learning_curve.csv`).

### Provenance rule

`data/processed/` is gitignored and lives on the cluster. So every result folder
**must name the paths** it depends on (config, processed dir, feature parquet,
benchmark output dir) in its README table — the artifacts themselves are not
committed, only the pointers to them.

## Keeping the index current

When you add or change a result, update:
- **`README.md`** (top-level) — the result table + the "Planned" list.
- **`history.md`** — the narrative thread, if the result changes the story.

## `status.md` — the living log (do not let work get lost)

**Every time Claude (or anyone) works on an experiment, append a log entry to that
folder's `status.md` before ending the session** — what was done, what's running,
what's blocked, and what the next step is. This is the mechanism that stops a
half-finished run (e.g. "features done, eval never ran") from being forgotten.

`status.md` template:

```markdown
# Status — NN_slug

**State:** 📋 Planned | 🚧 In progress | ⛔ Blocked | ✅ Done
**Last updated:** YYYY-MM-DD

## Current state
One short paragraph: where things actually stand right now (what exists on the
cluster, what's been run, what the headline number is if there is one).

## Next steps
- [ ] concrete next action (with the command / config / path to use)
- [ ] …

## Blockers
- (only if blocked) what's blocking, and how to retry / work around it.

## Log — newest first
### YYYY-MM-DD — <one-line what happened>
- detail, job IDs, paths, numbers, decisions.
```

Conventions:
- **Newest entry on top** of the Log.
- Log **absolute dates** (convert "today"/"last week").
- Record **job IDs, node names, and paths** — they're what the next session needs
  to resume or diagnose (e.g. a predict shard dying on `nodo3`/`nodo5`).
- Update **State** and **Last updated** on every touch.
- When an experiment finishes, set State to ✅ Done and make sure the README's
  headline numbers and the top-level index/`history.md` are updated to match.
