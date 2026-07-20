# Status — 09_external_benchmarks

**State:** 🟡 Design locked, not yet run. Test the raw-Δz/concat ΔΔG predictor on the
field-standard **external blind benchmarks S669 + Ssym**, under three training regimes
(Tsuboyama-only, FireProt-only, Tsuboyama→fine-tuned-on-FireProt), each scored **full**
and **homology-filtered** (leakage control).
**Last updated:** 2026-07-19

## Question
How does our Boltz-embedding ΔΔG predictor compare to the published literature on the
two most widely used blind benchmarks, and how much of any apparent skill is train↔test
sequence-identity leakage? Concretely, per benchmark:
- **Regime A — Tsuboyama-only:** train on all Tsuboyama features, predict the benchmark.
- **Regime B — FireProt-only:** train on all FireProt ≤500 features, predict the benchmark.
- **Regime D — Tsuboyama→fine-tuned:** pretrain on Tsuboyama, warm-start continue on
  FireProt (the 08 recipe), predict the benchmark.
Each scored **full** (comparable to papers) and **homology-filtered** (honest number).

Builds directly on 05 (cross-dataset transfer) and 08 (fine-tune A/B/D). The *only*
genuinely new/expensive work is extracting Boltz features for the benchmark mutations;
the training/scoring machinery already exists.

## Benchmarks (locked: S669 + Ssym)
- **S669** — 669 single-point variants / 94 proteins (Pancotti et al. 2022, *Brief.
  Bioinform.* bbab555). The de-facto blind test set; ThermoMPNN ≈ **0.55 Pearson**, RaSP
  ~0.39, DDGun/ACDC-NN ~0.4–0.5. Curated to ≤30 % identity to *its* reference training
  sets (S2648/VarBench) — but **not** necessarily to Tsuboyama/FireProt, which is why we
  add our own leakage control.
- **Ssym** — 342 direct + 342 reverse mutation pairs (Pucci et al. 2018), crystal
  structures for WT and mutant. Tests **antisymmetry** (pred(reverse) ≈ −pred(direct)) —
  directly relevant because 07/08 use antisymmetry augmentation. protddg-bench hosts a
  634-variant "SSYM" with 5-fold splits.

**Size cap ≤500 aa** (reuse the disk-bounded incremental-slim pipeline). Report benchmark
coverage (how many variants/proteins survive the cap).

## Leakage control (locked: report BOTH full + homology-filtered)
Field practice, and exactly what ThermoMPNN does: *"any homologues (>25 % sequence
identity) detected in the Megascale [=Tsuboyama] training set were removed prior to
retraining."* We replicate this at the **25 %** threshold (matches ThermoMPNN for a direct
comparison), and also report 30 %.

Method (reuse `ddg.evaluation.cluster`, MMseqs2):
1. Pool WT sequences of **{all Tsuboyama train proteins} ∪ {all FireProt proteins} ∪
   {benchmark proteins}** into one FASTA.
2. Cluster at 25 % (and 30 %) identity.
3. A benchmark protein is **leaky** if it shares a cluster with any *training* protein
   (Tsuboyama for regimes A/D, FireProt for regime B, either for the combined view).
4. **Full** = score all benchmark variants. **Filtered** = drop leaky benchmark proteins.
   The full→filtered drop *is* the leakage measurement; report it per regime × benchmark.

## Features & models (match 07/08 defaults)
- **Features:** concat `wtz_*` + `mtz_*` (128+128) from `features_ablation.parquet`
  (built by `results/07_feature_symmetry_ablation/build_ablation_features.py`; needs only
  the z-derived slim store, so `keep_s: false` is fine).
- **Model:** 5-seed MLP ensemble `MLPRegressor((256,128,64))`, **antisymmetry augmentation**
  on every training set (swap wtz|mtz halves, negate ddg) — identical to `run_finetune.py`.
- **Fine-tune (D):** warm-start the Tsuboyama-pretrained members on FireProt, reusing the
  Tsuboyama imputer/scaler (08 mechanism).
- **Metrics:** pooled r/ρ/RMSE/MAE + per-protein distribution. For Ssym additionally report
  the **antisymmetry** diagnostic (direct vs reverse prediction correlation / bias).
- **Sign convention:** benchmark ddG sign may differ; auto-flip on negative pooled Pearson
  (as `transfer.py` does) and record the flip.

## Data provenance (DONE — CSVs built + validated)
- **S669** → `data/raw/s669.csv`. Source: **DDGemb** (Bologna lab, the S669 authors'
  own canonical mapping): `S669.tsv` + `S669.fasta` from
  https://ddgemb.biocomp.unibo.it/static/ddgemb/data/. POS indexes the provided full-length
  UniProt sequence — **all 669 validated** `seq[pos-1]==WT`, 0 position errors (19 rows had a
  merged `<UniProt>WT<mut>` key, recovered by regex). **≤500 cap → 541 variants / 62 proteins**
  (dropped 128 variants in 25 large proteins). *Not* the DDGemb version's alternative: the
  ThermoMPNN PDB-construct S669 lacks sequences and reconstructing from RCSB SEQRES lost 123
  variants to author-numbering gaps — DDGemb is cleaner and authoritative.
- **Ssym** → `data/raw/ssym.csv`. Source: **ThermoMPNN** `data_all/testing/ssym-5fold_clean_dir.csv`
  (has a `SEQ` column). Positions are PDB-author-numbered → solved a **unique constant offset
  per protein**; kept only proteins with an unambiguously determined offset. **337 direct
  variants / 13 proteins** (all ≤500, seqlen 58–164). Dropped 2 offset-ambiguous proteins:
  1AMQA (4 muts all at one position on a `CC` motif, offset −10/−11 tie) and 1RN1C (1 mut,
  0/−5 tie). Direct only; reverse handled analytically via the antisymmetry model (see below).
- **Validated** through the real `minimal` adapter: both load with **0 position mismatches**.

## Plan / task list
- [x] **Acquire + normalize benchmark CSVs** → `data/raw/{s669,ssym}.csv` (`minimal` schema:
      `uniprot`, `mutation`, `wt_sequence`, `ddg`). See "Data provenance" above.
- [x] **Configs** `experiment_configs/s669.yaml`, `experiment_configs/ssym.yaml`
      (`dataset_type: minimal`, `mutate_wt_msa`/`mutate_first_row`, `max_msa 1000`,
      `keep_s: false`, `delete_raw: true` — mirror `fireprot_le200.yaml`). Written.
- [~] **Cluster feature extraction** (GPU, sharded, exclude bad nodes): SUBMITTED
      2026-07-19 as `prepare → predict[self-slim] → features` chains (jobs below). Still
      need `build_ablation_features.py` → `features_ablation.parquet` after each `features`.
- [x] **Homology map** — `build_homology_map.py` (MMseqs2 `easy-cluster` -c 0.8 via the
      project `cluster_wt_sequences`), pooled Tsuboyama+FireProt+benchmark WTs at 25 %/30 %.
      Written to `homology/{s669,ssym}_leakage.csv`. **Result:** S669 & Ssym have **0**
      proteins homologous to Tsuboyama; FireProt overlaps **S669 9 prot/181 var**,
      **Ssym 9 prot/290 var** (at 25 %). So regime A is a clean external test; regime B/D
      numbers are inflated by FireProt overlap — the filtered columns remove it.
- [x] **Scoring script** `run_benchmarks.py` — trains A/B/D once (concat + antisymmetry,
      5-seed MLP), predicts S669 & Ssym, scores full + filt25/filt30, Ssym antisymmetry
      diagnostic, auto sign-flip. Ready; needs the benchmark `features_ablation.parquet`.
      Local smoke test (train-only, benchmarks skipped) validates the training path.
- [ ] README + report.pdf (paper-facing) once numbers land; per-benchmark table vs literature.

## Open items / risks
- **S669 WT-sequence provenance:** positions in the paper are PDB-numbered; must be
  remapped to a 1-based index into the exact WT sequence we feed Boltz. This is the main
  data-wrangling risk — validate `wt_sequence[pos-1] == WT` for every row before extraction.
- **Ssym reverse mutations:** decide whether to feed reverse mutations (needs the mutant as
  "WT") or score antisymmetry from direct-only predictions. Leaning: extract **direct only**,
  derive the reverse prediction analytically via the antisymmetry-augmented model.
- Feature extraction is the long pole (cluster GPU). Everything else is local + cheap.

## Log — newest first
### 2026-07-19 — first submission failed (MSA rate-limit + bad node); resubmitted
- **First chains (212491–212496) died at `prepare`:**
  - **s669** (212491): ColabFold MSA server **rate-limited** — `50/62 MSAs failed` (12
    fetched). Documented failure mode; rerun prepare retries only the missing ones.
  - **ssym** (212494): prepare **actually completed** ("pipeline completed... 350 MSA
    files") but the job exited **127** because it landed on **nodo3** (ld.so teardown
    crash). Lesson: **`--exclude=nodo1,nodo3,nodo5` on the CPU steps too**, not just predict
    (submit_all.sh / cpu_step.sbatch don't exclude by default — I now pass it manually).
  - Cancelled the DependencyNeverSatisfied predict/features zombies.
- **Resubmitted with `--exclude` on every step:** ssym full chain prepare **212498** →
  predict **212499** → features **212500** (prepare skips the 350 cached MSAs → fast exit);
  s669 prepare-only retry **212501** (may need several passes until 0 MSAs fail, THEN chain
  predict→features). ssym MSAs are cached so it won't compete for the server.

### 2026-07-19 — homology map done; scoring script written; cluster jobs submitted
- **Homology/leakage map** (`build_homology_map.py`, MMseqs2 easy-cluster -c 0.8):
  S669 0 leaky vs Tsuboyama (both 25/30 %); 9 prot/181 var leaky vs FireProt (25 %),
  8/175 (30 %). Ssym 0 leaky vs Tsuboyama; 9 prot/290 var leaky vs FireProt (both).
  → `homology/{s669,ssym}_leakage.csv`. (First attempt used a Biopython identity/min(len)
  metric with no coverage filter — flagged everything, wrong; switched to MMseqs2 -c 0.8.)
- **`run_benchmarks.py`** written (A/B/D, full + filt25/filt30, Ssym antisymmetry, sign-flip).
- **Cluster jobs submitted** (git synced to f9ed096; chains = prepare→predict[self-slim]→features,
  `--exclude=nodo1,nodo3,nodo5`, %2 GPUs):
  - **s669**: prepare **212491** → predict **212492** (array 0-31) → features **212493**
  - **ssym**: prepare **212494** → predict **212495** (array 0-15) → features **212496**
- **Next:** after each `features` completes, run `build_ablation_features.py` (→
  `features_ablation.parquet`) on cluster, pull the two parquets locally, run `run_benchmarks.py`.
  Watch for MSA-server rate limits in prepare (75 fresh WT MSAs) and bad-node predict crashes.

### 2026-07-19 — benchmark data acquired, normalized, validated; configs written
- **S669** from DDGemb (S669 authors' UniProt mapping): all 669 validated, ≤500 →
  **541 variants / 62 proteins** → `data/raw/s669.csv`.
- **Ssym** from ThermoMPNN direct + unique per-protein offset: **337 variants / 13 proteins**
  (dropped 1AMQA, 1RN1C ambiguous) → `data/raw/ssym.csv`.
- Both validated through the `minimal` adapter (0 position mismatches). Configs
  `experiment_configs/{s669,ssym}.yaml` written. Build scripts kept in scratchpad
  (`build_s669_ddgemb.py`, `finalize.py`).
- **Next:** cluster feature extraction (prepare→predict→slim→features) for both, then
  homology map + scoring.

### 2026-07-19 — experiment framed & design locked
- Chose S669 + Ssym; leakage = report full + homology-filtered at 25 %/30 % (matches
  ThermoMPNN's Megascale >25 % removal); size cap ≤500 aa. Confirmed the training/scoring
  machinery already exists (transfer.py, run_finetune.py, cluster.py); only benchmark
  feature extraction is new. Scaffolded `results/09_external_benchmarks/`.
