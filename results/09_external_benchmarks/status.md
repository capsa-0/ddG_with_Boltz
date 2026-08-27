# Status — 09_external_benchmarks

**State:** ✅ Done. Both benchmarks extracted (S669 541/62, Ssym 337/13, full coverage),
scored (A/B/D × full/filtered/common), written up (README + figure). Optional: report.pdf.
**Last updated:** 2026-08-27

## Results (pooled Pearson r; per-protein median in parens)
| Benchmark | A Tsuboyama (leak-free) | B FireProt full → filt25 | D finetune full → filt25 |
|---|---|---|---|
| **S669** (541/62) | 0.255 (med 0.46) | 0.500 → 0.404 (med 0.58→0.56) | 0.462 → 0.408 (med **0.61**) |
| **Ssym** (337/13) | 0.728 (med 0.73) | 0.891 → 0.871 [n=47] (med 0.89→0.72) | 0.797 → 0.864 [n=47] (med 0.71) |

- **Leakage confirmed:** A (Tsuboyama) has 0 overlap → full==filtered on both. B/D drop when
  homologs removed (S669 B 0.50→0.38–0.40; Ssym B per-prot med 0.89→0.72).
- **S669 = the honest hard test** (diverse); pooled r modest (0.26–0.50). **Ssym = easy/narrow**
  (lysozyme/barnase) → inflated for everyone.
- **Distribution > size:** FireProt (natural proteins) beats Tsuboyama (designed domains) on
  S669 even filtered. **Fine-tune (D)** has the best S669 per-protein median (0.61) and it holds
  under filtering — contrasts exp 08's within-FireProt "washes out".
- **Antisymmetry** (Ssym): corr(dir,-rev) 0.91/0.98/0.97 for A/B/D, bias ~0.05.
- All benchmarks sign-flipped (opposite ΔΔG convention; auto-handled).

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

### 2026-08-27 — report.pdf + segunda figura, y **el README estaba desactualizado**

**Hallazgo que importa.** El banner de corrección del README decía "Every number below is
the corrected one" — **y no lo era**. Las dos tablas de resultados (S669 y Ssym), la línea de
antisimetría y los hallazgos 2–4 seguían con los valores **pre-corrección**, idénticos a
`results_pre-correction.csv`. Solo la sección "What the correction changed" tenía los
corregidos. Es decir: quien leyera el README se llevaba los números defectuosos con un cartel
que le aseguraba lo contrario.

Corregido contra `results.csv`. Los cambios que importan:

| | README decía | real (corregido) |
|---|---|---|
| S669 A full | 0.255 (0.46) | **0.415 (0.55)** |
| S669 A common25 | 0.214 (0.48) | **0.361 (0.59)** |
| S669 B filt25 | 0.404 (0.56) | 0.460 (0.59) |
| S669 D filt25 | 0.408 (0.61) | 0.453 (0.54) |
| Ssym B full | 0.891 (0.89) | 0.850 (0.85) |
| antisimetría A/B/D | 0.91/0.98/0.97, "bias ≈ 0.05" | 0.945/0.979/0.973, bias **+0.25/+0.04/−0.23** |

**Una conclusión se da vuelta.** El hallazgo 4 decía "fine-tuning earns its keep on the hard
benchmark: regime D has the best S669 per-protein median (0.61)". Con los números corregidos
**D es el último** en mediana por proteína (0.54 contra 0.59 de A y B) y está por debajo de B
en r filtrado (0.453 vs 0.460). Reescrito: fine-tuning **no** rinde acá — lo que además ahora
concuerda con results/08, donde también se lavaba.

También revisado el hallazgo 3: la brecha B−A en common25 pasó de 0.190 a **0.099**, la mitad.

**Figura nueva** `02_correction_and_antisymmetry.png`: (a) el efecto de la corrección por
benchmark × subset × régimen, con la flecha antes→después — la ganancia sigue el tamaño del
corpus de entrenamiento (A +0.15/+0.16 contra +0.02/+0.06 de B y D), que es la firma que
predice el argumento de las épocas; (b) el sesgo residual de antisimetría en Ssym, que es
**dependiente del régimen** y estaba subestimado en el README.

**`report.pdf`** construido (3 páginas, `build_report.py`, números leídos de los CSV).
Verificado: 2 figuras embebidas, 0 términos de procedencia. La figura 01 ya estaba
regenerada post-corrección (01:26 contra 01:25 del CSV), confirmado.

**`figures/README.md`** creado — no existía, y las guidelines lo piden. Incluye la advertencia
de que las barras "filtered" de `01` **no son comparables entre regímenes** (cada una descarta
un conjunto distinto de proteínas); solo `common25` sostiene una afirmación cruzada.

### 2026-08-27 — regime A is overfit; its S669 number is understated (found in results/14)

`run_benchmarks.py:65` builds the MLP members with **`early_stopping=False, max_iter=250`**,
unlike the project default `make_model("mlp")` (`early_stopping=True, max_iter=1000,
n_iter_no_change=25, validation_fraction=0.1`). Isolating that single change — same data,
same concat features, same antisymmetry, same 5-seed ensemble, same 541-variant S669:

| estimator | S669 pooled r | ρ |
|---|---|---|
| `early_stopping=False, max_iter=250` (as published here) | **0.255** | 0.262 |
| `early_stopping=True, max_iter=1000` (project default) | **0.415** | 0.440 |

The first reproduces this folder's published 0.255 / 0.262 exactly, so the difference is
purely the estimator. **Regime A's S669 result is understated by ~0.16 Pearson.**

**This may affect the study's central comparison.** `max_iter` is epochs, so at a fixed 250
with no early stopping, regime A (12,359 muts → 24,718 augmented) received ~4x the gradient
updates of regime B (3,205 → 6,410): the regime with the most data was overfit hardest. The
reported ordering on the homology-controlled subset (A 0.214 vs B 0.404, D 0.408) is
therefore confounded and should not be read as "Tsuboyama-only transfers worse to S669"
until re-run.

**FIXED the same day.** `members()` now uses the project-default estimator
(`early_stopping=True, max_iter=1000, n_iter_no_change=25, validation_fraction=0.1`;
`warm_start` retained for regime D's fine-tune, which now early-stops too). All three
regimes re-run on CPU locally, ~25 min — no GPU and no re-prediction needed, since every
feature table was already on disk. Originals archived as `results_pre-correction.csv` and
`ssym_antisymmetry_pre-correction.csv`; `results.csv`, README and
`figures/01_pooled_r_full_vs_filtered.png` regenerated.

| benchmark | subset | A Tsuboyama | B FireProt | D fine-tuned |
|---|---|---|---|---|
| S669 | full | 0.255 → **0.415** | 0.500 → 0.546 | 0.462 → 0.506 |
| S669 | common25 | 0.214 → **0.361** | 0.404 → 0.460 | 0.408 → 0.453 |
| Ssym | full | 0.728 → 0.759 | 0.891 → 0.850 | 0.797 → 0.780 |

Regime A gains +0.15/+0.16, B and D only +0.04/+0.06 — the asymmetry the epoch-count
argument predicts. **The ordering survives**: FireProt-trained still leads pooled r on the
honest common-25 subset. **But two things changed materially.** (i) The corpus effect on
S669 roughly halves (common-25 gap B−A: 0.190 → 0.099). (ii) **The per-protein median r is
now tied, A 0.594 vs B 0.594** — Tsuboyama-only ranks within-protein as well as
FireProt-only, and lags only on the cross-protein offset, i.e. the domain-shift term from
results/11. The README's "distribution > size" claim has been rewritten accordingly: it is
a calibration advantage, not a ranking one.

### 2026-07-20 — both benchmarks extracted (full coverage) and scored
- **ssym predict (212600) recovered to 337/13** — the slim-clobber fix worked (was 165/10).
- **s669 predict (212602) completed 541/541 var, 62/62 prot, 0 failures** across ~2.5 h with
  no_kernels + small-GPU excludes. Pulled both slim stores, built `features_ablation.parquet`
  (ssym 337, s669 541, 0 skipped), ran `run_benchmarks.py` → `results.csv` (18 rows) +
  `ssym_antisymmetry.csv`. Headline in Results block above.
- Added a `common{thr}` subset (all regimes on the leaky_any-clean variants) for a fair
  cross-regime comparison; re-running scoring. **Next:** paper-facing README + report.pdf
  (+ scatter/leakage figures), then commit results.

### 2026-07-20 — two more bugs: small-GPU OOM + slim resume-clobber; both fixed; resubmitted
- **`--no_kernels` confirmed working** (ssym shards ran fine on nodo11, the node that crashed before).
- **s669 predict OOM'd:** all its shards landed on **nodo12**, whose GPU is only **~1.94 GiB**
  (nodo5 & nodo12 are the small 46900-RAM nodes; the rest are 62266). Fix: also exclude the
  tiny-GPU nodes → `--exclude=nodo1,nodo3,nodo5,nodo12,sauron`. (Disk is tight — shared NAS at
  100%, ~193 GB free — so keep `delete_raw: true`; can't afford to keep raw.)
- **ssym came out incomplete (165/337 muts, 3 proteins gone) — a slim resume bug.** Shards
  s0000/s0005/s0010 (the 3 that succeeded in run 1) had **0 structures**: on the run-2 resume,
  predict skipped their already-slimmed structures, then slim wrote an **empty shard, clobbering
  the good one**, erasing the WT structures of 1BNIA/1L63A/1OH0A + scattered mutants. The
  NFS "Device or resource busy" tracebacks in the logs are unrelated noise from Boltz's
  multiprocessing exit cleanup (non-fatal — shards wrote fine after them). **Fix (6b4acea):**
  `slim` now never overwrites an existing shard with an empty one.
- **Resubmitted (git 6b4acea, no_kernels + node excludes + slim fix):** ssym predict **212600**
  (resume regenerates the 3 empty shards) → features **212601**; s669 predict **212602** (fresh)
  → features **212603**.

### 2026-07-20 — predict failed on cuequivariance kernel; fixed with --no_kernels; resubmitted
- Both `prepare` steps eventually **COMPLETED** (s669 586 MSAs, ssym 350) after the retries.
- **ssym predict (212499) failed:** 3/16 shards (0,5,10, all on **nodo8**) succeeded; the
  rest (on **nodo11**) crashed with `ModuleNotFoundError: No module named 'cuequivariance_torch'`
  inside Boltz's compiled `triangular_mult` kernel. Root cause: the package is **not installed**;
  Boltz enables its optimized triangular-mult kernel by **GPU architecture**, so nodo11's GPU
  takes the cuequivariance path (crash) while nodo8's falls back to pure torch (works). Not a
  simple bad-node issue and not fixable by excluding one node (other kernel-capable nodes would
  fail too).
- **Fix:** added an opt-in `boltz_flags.no_kernels` → passes Boltz `--no_kernels` (forces the
  pure-torch path on every node; `ddg/feature_extraction/extraction/run_boltz.py`, commit 40423fa).
  Enabled in both configs. This matches how the Tsuboyama/FireProt training features were made
  (cuequivariance was never installed, so those ran pure-torch too). Slight slowdown, negligible
  for these small proteins; numerically the same operation.
- **Resubmitted (git 40423fa):** ssym predict **212545** (resumable, skips slimmed 0/5/10) →
  features **212546**; s669 predict **212547** → features **212548**. All `--exclude=nodo1,nodo3,nodo5`.

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
