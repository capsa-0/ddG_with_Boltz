# TODO — refactors & stress tests

Planning/working notes, tracked in the repo. Captures decisions made
2026-07-16/17 (and later) so we can execute without re-deriving context.

**Current state (context):**
- The **raw-Δz representation** is the decided feature set: `Δz` diagonal (128) +
  `Δz` mean-pooled row (128) = 256 features. `s` is **optional** (redundant with raw
  z at the model level; keep as a config switch only).
- Raw-Δz features currently produced by an **ad-hoc script** (`build_rawz.py` on the
  cluster), *not* the pipeline. The pipeline's `features` step still runs the old
  **summary-statistics** extractor (`ddg/exploration/`).
- Data on cluster: `data/processed/tsuboyama_bench_{fast,wide}/` — `slim/` +
  `rawz_features.parquet` + `mutations.csv` + benchmark outputs. (msas/queries/
  exploration_plots deleted.) Local copy of `fast` in `data/processed/`.
- Model: `HistGradientBoosting` (`ddg/evaluation/`). Benchmark/eval module stays.

---

## 1. Code refactor — raw-Δz feature pipeline, remove summary/exploration

Goal: the pipeline's `features` step emits **raw Δz [+ optional Δs]** directly; delete
the summary-statistics + old-plots code.

- [ ] **New feature builder** `ddg/features/build_features.py` (or reuse
      `ddg/feature_extraction/`): read the slim store + `mutations.csv`, and per
      mutation at position `i` emit:
  - `zdiag_0..127` = `mut_z[i,i] − wt_z[i,i]`
  - `zpool_0..127` = mean over residues of `mut_z[i,:] − wt_z[i,:]`
  - `sdim_0..Ds-1` = `mut_s[i] − wt_s[i]` **only if** `slim.keep_s` (config switch)
  - metadata: `wt_id, mutation, ddg`; inf→NaN. Write `features_summary.parquet`.
  - (Reference impl: the cluster `build_rawz.py` — port it into the module.)
- [ ] **Wire the pipeline**: `ddg/pipeline.py::_run_features` → call the new builder
      instead of `ddg.exploration.explore_features`.
- [ ] **Update the eval ablation**: `ddg/evaluation/labels.py` →
      `S_FEATURE_PREFIXES = ("sdim_",)` so `--drop-s` still works on the new columns.
- [ ] **Delete** `git rm -r ddg/exploration/` (explore_features.py,
      feature_analysis/{extractors,feature_analyzer,plots}.py — summary stats, UMAP,
      correlation, scatter).
- [ ] **Check dead code**: `ddg/datasets/boltz_dataset.py` (raw-NPZ reader) is now
      unused (slim store is the source; raw NPZs deleted). Remove or keep — decide.
- [ ] **slim default**: `keep_s: false` as the sensible default (z-only) once raw Δz
      is the pipeline; `keep_s: true` only when experimenting with s.
- [ ] **Test** locally on `data/processed/tsuboyama_bench_fast/slim`: new features
      step → `features_summary.parquet` should reproduce `rawz_features.parquet`
      (same 256 z columns); with `keep_s` it adds `sdim_*`.
- [ ] **Update `CLAUDE.md`** repo-layout: `exploration/` → `features/`; features step
      now raw Δz.
- [ ] Commit (this is a repo change → git rm + new module).

---

## 2. Results refactor — one folder per experiment/result

Goal: `results/` holds a folder per result, each self-contained (README + figures +
report). Currently `results/` is flat (the generalization study at top level).

Proposed layout:
```
results/
  README.md                     # index of all results
  01_generalization/            # the current raw-Δz report (report.pdf + details.md)
      README.md  report.pdf  details.md  figures/
  02_stress_extrapolation/      # tail extrapolation (§3 stress tests)
      README.md  report.(pdf|md)  figures/
  03_stress_learning_curve/
      README.md  report.(pdf|md)  figures/
  04_cross_dataset_fireprot/    # once FireProt runs
      README.md  report.(pdf|md)  figures/
```

- [ ] `git mv` current `results/{report.pdf, details.md, figures/}` →
      `results/01_generalization/`.
- [ ] Write `results/README.md` as the top-level index (one line per experiment +
      headline number).
- [ ] Each experiment folder: `README.md` (what/why/how + key numbers), `figures/`
      (numbered PNGs + a figure index), `report.(pdf|md)`.
- [ ] Fix figure paths in the moved report if rebuilt.
- [ ] Commit (results/ is tracked; push).

---

## 3. Stress tests (decided) — run on the WIDE corpus (raw Δz)

Data: `data/processed/tsuboyama_bench_wide/rawz_features.parquet` (ready, 37 k
mutations, 256 raw-Δz features). Model: HGB. Run on the cluster.

- [ ] **Extrapolation to the destabilizing tail.** Train only on **mild** mutations
      (|ΔΔG| < 1 kcal/mol), test on the **strongly destabilizing tail** (ΔΔG > 2).
      Report r / RMSE / MAE and the fit slope on the tail vs. the in-distribution
      baseline. Targets the regression-to-the-mean weakness already observed (fit
      slope < 1). → figure: predicted-vs-actual on the tail + a slope/coverage stat.
- [ ] **Learning curve.** Train on 10 / 25 / 50 / 100 % of *proteins* (grouped, so
      test proteins stay unseen), plot pooled r vs. #training proteins. Shows data
      efficiency / whether it's saturated. → figure: r (and RMSE) vs. training size.
- [ ] Package both into `results/02_stress_extrapolation/` and
      `results/03_stress_learning_curve/` per §2.

### Blocked / later
- [ ] **Cross-dataset stress — FireProt** (`experiment_configs/fireprot_le200.yaml`,
      85 proteins / 1,543 muts ≤200 aa). **Blocked**: ColabFold MSA server rejecting
      all requests (rate-limited by heavy use today). Retry `prepare` in a few hours
      (resumable); if the server stays down, options = local MMseqs2 on the cluster,
      or single-sequence Boltz (confounds the comparison). Then predict → slim (keep_s
      false) → extract raw Δz → test the **Tsuboyama-trained** model on it. **Watch the
      ddG sign convention** (FireProt vs Tsuboyama positive=destabilizing) — flip if
      the correlation comes out negative.
- [ ] (Optional) **Compound holdouts** — de-novo + leave-one-substitution, or
      homology-cluster + chemistry. Cheap re-splits, strictly harder than single-axis.
- [ ] (Optional) **Double-mutant epistasis** — does summed single-mutant raw-Δz
      predict Tsuboyama doubles? Needs the double-mutant embeddings.

---

## 4. New benchmark — "Does our stability signal predict function?" (RF4Mave)

Goal: test whether our Boltz-embedding ΔΔG carries stability information competitive
with Rosetta ΔΔG for predicting **MAVE functional fitness** (not ΔΔG accuracy — that's
Tsuboyama's job). This is a *stability→function transfer* / generalization test against
a published, well-defined baseline. Reproduces Høie et al. 2022 (Cell Reports 38,
110207, "RF4Mave") with our ΔΔG swapped in for Rosetta's.

**Paper:** Høie, Cagiada, Frederiksen, Stein, Lindorff-Larsen (2022),
"Predicting and interpreting large-scale mutagenesis data using analyses of protein
stability and conservation." PDF: `theory/biblio/marce/RF4Mave.pdf`.
DOI: 10.1016/j.celrep.2021.110207.

**Where to find the dataset (public):**
- Zenodo: https://doi.org/10.5281/zenodo.5647207  (record `5647208`)
- GitHub: https://github.com/KULL-Centre/papers/tree/main/2021/ML-variants-Hoie-et-al
- Dataset list per protein: Table S1 of the paper.
- Content: 39 MAVE datasets over **29 proteins**, 154,808 single-AA variants at 10,012
  positions. Per variant they provide the triplet **(s_exp, Rosetta ΔΔG, GEMME ΔDE)**:
  - `s_exp` = rank-normalized MAVE fitness (1 = WT-like activity, 0 = loss of function) — the **target**.
  - Rosetta ΔΔG (kcal/mol) = stability feature — **this is what we replace with our Boltz ΔΔG**.
  - GEMME ΔDE = conservation feature (rank-normalized 0–1).

**Setup to reproduce (their exact protocol):**
- **Leave-one-protein-out** cross-validation across the 29 proteins (when a protein has
  multiple MAVEs, exclude the extras during training; evaluate on all of its MAVEs).
- Model: random forest (their choice; robust, few hyperparams). We can also try our HGB
  for consistency, but report RF to match them.
- Metric: **median Spearman ρ** between predicted and experimental fitness across the 39
  datasets (expect a broad per-protein spread, ρ ≈ 0.1–0.8, so median is the headline).

**Baselines to beat (their published numbers, median Spearman ρ, LOPO):**
- ΔΔG-only (Rosetta): **0.25**
- ΔDE-only (GEMME): 0.42
- ΔΔG + ΔDE combined: **0.47**
- Full "position-context" model (all 20 ΔΔG + 20 ΔDE at the position + s̃ features): **0.52**
- Null (substitution-matrix `s̃_exp`): 0.17
→ Headline claim if it works: our-ΔΔG-only beats/≈ Rosetta-ΔΔG-only (0.25), and our-ΔΔG+ΔDE
  matches/beats 0.47.

**Work items:**
- [ ] Download the Zenodo/GitHub data; confirm per-variant (s_exp, ΔΔG, ΔDE) tables +
      the WT sequences/structures + MSAs are usable. Map their 29 proteins → WT FASTA.
- [ ] Decide scope: full 29 proteins is a **large Boltz run** (154 k variants). Consider
      a subset first (e.g. proteins ≤200 aa, or the ~6 with the cleanest MAVE↔ΔΔG signal).
- [ ] New `ddg/datasets/` adapter for their per-protein variant tables (register in
      `load_input_dataset.py`) — keep it behind an adapter, don't special-case downstream.
      Target column is `s_exp` (function), NOT ddG — this benchmark predicts fitness, so
      the eval target differs from the rest of the pipeline; keep our ΔΔG as a *feature*.
- [ ] Run the pipeline (prepare→predict→slim→features) to get our Boltz ΔΔG per variant.
      (We predict ΔΔG from raw-Δz first, then feed it as the stability feature — mirrors
      how Rosetta ΔΔG is a scalar input to their RF.)
- [ ] Reproduce their RF (LOPO, median Spearman) with: (a) our-ΔΔG only, (b) our-ΔΔG +
      GEMME ΔDE (reuse their ΔDE column), (c) optional position-context features.
- [ ] Package into `results/05_rf4mave_function_transfer/` (per §2): README with the
      table above (their numbers vs ours) + per-protein Spearman boxplot (their Fig 2B style).

**Caveats / watch-outs:**
- Target mismatch is the whole point: fitness ≠ ΔΔG, so even a perfect ΔΔG predictor
  caps out well below ρ=1. Don't read low ρ as a failure of our ΔΔG — read it *relative
  to Rosetta's 0.25*.
- Their ΔΔG column is **Rosetta-predicted**, not experimental — use it only as the
  baseline-to-beat, never as ground truth.
- Reuse their GEMME ΔDE column rather than recomputing conservation, to isolate the
  effect of swapping the stability term.

---

## 5. Dataset provenance & audit files (explain how each dataset was derived)

Goal: produce **clean, auditable documentation** for every dataset we feed the
pipeline, so a colleague (or reviewer) can trace each row from the original database
dump -> the filtered CSV the pipeline consumes, and reproduce the filtering. Right now
the derivations live only in ad-hoc, undocumented notebooks (Spanish, no markdown,
exploratory) under `ddg_datasets/` — the "how did we get this dataset" is not written
down anywhere durable.

**What's in `ddg_datasets/` today (the raw material to document):**
- `dms/` — **Tsuboyama 2023 K50 megascale** (mega-DMS). Source download:
  `dms/Processed_K50_dG_datasets/.../Tsuboyama2023_Dataset1_20230416.csv` +
  `Dataset2_Dataset3_...csv` (raw K50/dG fits, ~40 cols incl. `deltaG`, 95% CIs,
  per-protease log10_K50). Cleaning notebook: `dms/dms.ipynb` (undocumented).
  Output the pipeline uses: `dms/tsuboyama_single_mutants_ddg.csv`
  (`protein_id, wt_sequence, mutation, ddg`; **389,069 rows**). The benchmark subsets
  (`tsuboyama_bench_fast/wide`) are then built by `ddg_datasets/build_benchmark_corpus.py`.
- `fireprot_cleaning/` — **FireProtDB**. Source dump: `fireprotdb_results.csv`
  (**53,445 rows**, 35 cols incl. `ddG, dTm, is_curated, method, pH, sequence`).
  Cleaning notebook: `direprot.ipynb` (dedup on `uniprot_id/chain/position/wild_type/
  mutation`, duplicate-variance checks — undocumented). Output:
  `fireprot_filtered.csv` (**3,662 rows**, adds `source_rows`). The <=200 aa slice
  `data/raw/fireprot_le200.csv` feeds `experiment_configs/fireprot_le200.yaml`.

**Deliverable — one audit doc per dataset** (proposed: `ddg_datasets/<name>/DATASET.md`,
committed; decide location — these should be *shared*, unlike `docs/`):
- [ ] **Source & citation** — where the raw dump came from (DB name, URL/DOI, download
      date, version), and the paper to cite.
- [ ] **Raw schema** — columns of the source file, units, and which ones we keep.
- [ ] **Filtering steps, in order** — every drop/dedup/threshold with the **row count
      before->after** each step (a reproducible funnel: 53,445 -> ... -> 3,662). Port the
      logic out of the notebook into prose (+ ideally a small deterministic script).
- [ ] **Output schema** — columns of the pipeline-facing CSV, the ddG **sign
      convention** (positive = destabilizing? FireProt `ddG` vs Tsuboyama `ddg` differ —
      see CLAUDE.md), and units.
- [ ] **Known caveats** — duplicate measurements, mixed assays/pH, curated vs derived,
      designed vs natural proteins, dTm-vs-dG entries, etc.
- [ ] **Sanity/audit checks** — n_proteins, n_mutations, WT-sequence consistency
      (mutation position matches the stated WT residue), ddg distribution + outliers.

**Approach:**
- [ ] Turn each cleaning notebook into a **deterministic script**
      (`ddg_datasets/<name>/clean_<name>.py`) that reproduces the filtered CSV from the
      raw dump — the audit doc references it so the funnel is re-runnable, not just
      described. (The existing `raw_vs_summary.py` is a separate untracked helper —
      review whether it belongs here.)
- [ ] Consider whether the cleaning belongs behind the dataset adapters in
      `ddg/datasets/cleaning/` (already exists, e.g. `make_tsuboyama_subset`) rather
      than standalone notebooks — align with the "new formats behind an adapter" rule.
- [ ] **Do NOT commit the large raw dumps** (`dms/` is 3.2 GB, `fireprotdb_results.csv`
      28 MB) — document their provenance/how to re-download instead; keep only the
      small pipeline-facing CSVs (or note where they live). Add a `.gitignore` rule for
      the bulky raw files if we start tracking this dir.

**Note:** `ddg_datasets/` is currently **entirely untracked** (nothing committed, not
ignored). Decide per-file what to commit (docs + small CSVs + scripts) vs. ignore (raw
dumps, `venv_datasets/`, `__MACOSX/`, `*.zip`, PNGs) as part of this task.
