# results v2 — supplementary details & methods

Everything that didn't fit the clean narrative of `results_v2.pdf`: exact
mechanics, hyperparameters, provenance, the numbers behind the summary statements,
and the caveats. Read the PDF first; this is the appendix.

---

## 1. Provenance & reproducibility

### Corpus
- **Dataset:** Tsuboyama et al. 2023 mega-scale stability dataset, single mutants
  (`data/raw/tsuboyama_single_mutants_ddg.csv`, 389k rows, 412 proteins).
- **This corpus (`tsuboyama_bench_fast`):** built by
  `ddg_datasets/build_benchmark_corpus.py --k 30` — all 412 proteins, ~30 mutations
  each (stratified), **12,359 mutations** after one drop (a key mismatch).
- **Wide corpus (`tsuboyama_bench_wide`, --k 90, ~37k):** predicted and slimmed on
  the cluster, but its features/holdouts still use the **summary** representation —
  it has **not** been re-run on raw Δz yet (see §7).

### Embeddings
- Boltz-2 run in **embeddings-only** mode (`--embeddings_only`), on the cluster via
  SLURM. Kept the trunk tensors `s` (L×384), `z` (L×L×128), `pdistogram` (L×L×bins)
  for wild-type and mutant.
- Compacted into a **slim store** (`ddg/storage/slim.py`): full `s`, plus only the
  mutation-position **row** of `z`/`pdistogram`. Originally float16; see the overflow
  note in §2.

### Scripts / where things live
| Artifact | Path |
|---|---|
| Corpus builder | `ddg_datasets/build_benchmark_corpus.py` |
| Summary features → parquet | `ddg/exploration/` (features step) → `features_summary.parquet` (653 cols) |
| Holdout suite | `ddg/evaluation/` (`python -m ddg.evaluation`) |
| Feature-representation experiment | `raw_vs_summary.py` (scratchpad) |
| Raw-Δz extractor | `build_rawz_parquet.py` (scratchpad) → `rawz_features.parquet` (256 cols) |
| v2 report | `docs/results_v2.md` / `.pdf`; figures in `docs/results_v2_figures/` |

*(The two scratchpad scripts are not committed; they read the slim store + `mutations.csv`.)*

---

## 2. Feature extraction — exact mechanics

At the mutated residue `i` (0-based), for WT and mutant:

- **`s[i]`** — single-track vector, 384-dim.
- **`z[i, i, :]`** — pair diagonal (residue with itself), 128-dim.
- **`z[i, :, :]`** — pair interaction row (residue with all L), L×128.

**Summary representation (original, 653 features).** Each slice's WT−mutant
difference is taken in three "modes" — `abs` = |Δ|, `signed` = Δ, `l2` = per-residue
L2 norm — then **flattened and reduced to ~14 scalar moments** (mean, SD, sum, max,
mean_abs, gini, entropy, skew, kurtosis, and normalized variants) by
`extractors._get_stats`. Plus 384 raw per-dim `Δs` and a cosine similarity. No raw
`z` survives — only its moments.

**Raw-Δz representation (this report, 256 features).**
- `Δz_diagonal` = `mut_z[i,i] − wt_z[i,i]` — 128 raw dims.
- `Δz_row-pooled` = `mean over residues of (mut_z[i,:] − wt_z[i,:])` — 128 dims
  (the row is variable-length in L, so it must be pooled).
- Concatenated → 256. No `s`, no summary stats.

**The float16 overflow.** The slim store cast `s` and `z` to float16; some values
exceed 65504 → `inf`/`NaN`, which propagated into features (imputed downstream).
Fixed for `s` (now float32, `slim.py`); **`z` is still float16**, so the raw-Δz
features here inherit a small amount of that corruption — the true raw-Δz numbers
are likely a touch *higher* than reported.

---

## 3. Model & hyperparameters

- **HGB** = `HistGradientBoostingRegressor`, in a pipeline
  `SimpleImputer(median) → StandardScaler → HGB`.
  - Holdout suite: `max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
    l2_regularization=1.0, early_stopping=True`.
  - Feature-representation experiment: same but `max_iter=300`.
- **SVR reference** (RBF, `C=10, gamma=scale, epsilon=0.1`): on the *summary*
  features it scored **higher** on the holdouts it could finish — random 0.758,
  protein 0.736, de-novo 0.630 (vs HGB 0.714 / 0.702 / 0.615) — but SVR is O(n²–³)
  and could not finish the leave-one-out sweeps (a single run took ~6 h and stalled),
  so HGB is the workhorse. **SVR was not re-run on raw Δz.**

---

## 4. Holdout implementation details

Defined in `ddg/evaluation/splits.py`:
- **random** — `KFold(10, shuffle)`.
- **protein / cluster** — `GroupKFold(5)` on `wt_id` (cluster needs an external map).
- **de-novo** — binary transfer: train natural→test designed, and reverse (2 folds).
- **substitution / source / target / chemistry** — `leave_out_folds`, one fold per
  category, **filtered to categories with ≥ 10 test and ≥ 50 train mutations**.
  → **332** of 380 substitutions qualify; **12** of 13 chemistry classes (C→X has 1
  example); all 20 source and 20 target residues.

**Metric — pooled vs. per-unit.** "Pooled r" = one Pearson over *all* out-of-fold
predictions. It differs from the mean of the per-category r's, because pooled also
rewards getting each category's *baseline* ΔΔG right. Example (substitution): pooled
**0.772** vs mean-of-332-cells **~0.53**. The report quotes pooled; per-category
figures (substitution heatmap, chemistry bars) show the per-cell values.

---

## 5. The feature-representation experiment — full table

Random **5-fold** CV, HGB, fast corpus (`raw_vs_summary.py`):

| Representation | # feat | CV r |
|---|---|---|
| z summary stats | 84 | 0.626 |
| pd summary stats | 86 | 0.639 |
| raw Δs | 384 | 0.658 |
| **ALL summary stats (original pipeline)** | 653 | **0.710** |
| concat WT‖mut s | 768 | 0.717 |
| **raw Δz diagonal** | 128 | **0.752** |
| raw Δz row-pooled | 128 | 0.768 |
| **raw Δz (diagonal + pooled)** | 256 | **0.780** |
| raw Δs + raw Δz | 640 | 0.774 |
| concat WT‖mut (s + z-diagonal) | 1024 | 0.781 |

Notes: adding raw Δs to raw Δz *slightly hurts* (0.780→0.774) → `s` is redundant with
raw `z`. Concatenation ≈ delta (0.781 vs 0.780) at 4× width. The §3 holdout numbers
(random 0.783) use the eval's **10-fold** random CV, hence marginally above the 0.780
here.

## 5b. Other numbers not in the PDF

- **s-ablation (summary features, HGB):** with-s vs no-s — random 0.714/0.679,
  protein 0.702/0.669, de-novo 0.615/0.579 (s adds ~0.03–0.05). This motivated
  keeping s under the *summary* representation — but §5 shows it's moot once z is raw.
- **Per-protein (raw Δz):** mean 0.806, median 0.831, SD 0.109; 87 % of proteins
  r>0.7, none <0.3, none <0.
- **Regression to the mean:** the predicted-vs-actual fit slope is < 1 — the model
  under-predicts the most destabilizing mutations (visible in figure 04).

---

## 6. What each v2 figure shows (`docs/results_v2_figures/`)

| File | Report § | Content |
|---|---|---|
| 01_feature_comparison_raw_vs_summary | §1.3 | raw vs summary representations, CV r |
| 02_holdout_generalization | §3 | raw-Δz Pearson r per holdout |
| 03_per_protein_distribution | §3.1 | per-protein r box/strip |
| 04_predicted_vs_experimental | §3.2 | pred-vs-actual hexbin (random/protein/substitution) |
| 05_substitution_heatmap | §3.3 | 20×20 source×target leave-one-substitution-out r |
| 06_chemistry_class_introduce_vs_remove | §3.3 | per-chemistry-class r, introduce vs remove |

---

## 7. Caveats & open work

- **Fast corpus only (12k).** The **wide corpus (37k)** is predicted + slimmed but
  its pipeline still uses **summary** features — it has *not* been re-run on raw Δz.
  The main outstanding task: **rebuild the extractor to emit raw Δz directly**, then
  re-run wide.
- **Cluster / homology holdout not run** — the one missing generalization axis. The
  identity-clustering attempt over-merged (single-linkage percolation) and needs a
  coverage-gated fix.
- **`z` still float16** in the slim store (only `s` was moved to float32), so raw-Δz
  numbers are marginally *under*-stated.
- **SVR not re-run on raw Δz** (it beat HGB slightly on summary features but doesn't
  scale to the leave-one-out sweeps).
- **The "introduce > remove" chemistry asymmetry** (figure 06) is an observation, not
  a validated claim — bucket composition and n differ across classes.
- No hyperparameter tuning / nested CV; HGB defaults are reasonable but untuned.

---

## 8. Related documents

- `docs/results_v2.md` / `.pdf` — the report this supplements.
- `docs/results.md` — v1 (summary features), with extra feature analysis not repeated
  here: **UMAP** of the feature space, correlation-by-family, and the feature-group
  ablation.
- `docs/experiment_log.md` — chronological log (corpus, runs, the transient-NFS slim
  failure + recovery, the SVR→HGB switch, the MSA-server reuse).
- `docs/benchmark_plan.md` — the full holdout methodology / roadmap.
