# Status — 16_aftoolkit_headtohead

**State:** ✅ Done — both corpora settled. S669 paired comparison + FireProt leakage reversal.
**Last updated:** 2026-08-28

## Current state

**S669 is finished and is a *paired* comparison, not a comparison against a published
aggregate.** AFToolkit ships its precomputed AF2 features for all 669 S669 variants and
its three trained adapters, so its own per-variant predictions were reproduced locally
(no GPU): SVM ρ 0.515 / r 0.518 / RMSE 1.414 against the paper's 0.51 / 1.41 — an exact
reproduction. Those predictions were mapped onto this project's variant ids and both
methods scored on the identical 541 variants / 62 proteins.

Result: this project's best transfer configuration (`zdiag`, 128 d, Tsuboyama-only)
scores **ρ 0.569 / r 0.557 / RMSE 1.357** against AFToolkit's **ρ 0.511 / r 0.525 /
RMSE 1.401** on those same variants. Paired protein-cluster bootstrap:
**Δρ +0.063 [−0.002, +0.144]** (P(ours ahead) = 0.97), **Δr +0.042 [−0.029, +0.133]**
(P = 0.84). Nominally ahead on every metric, *not* significant at 95 % — the honest
reading is parity, achieved with 128 features on a frozen trunk and 12,359 training
mutations against AFToolkit's 223,611.

The project's *adopted default* (concat, 256 d) is **behind** AFToolkit by
Δρ −0.130 [−0.226, −0.017] — significant. The parity is bought entirely by the
transfer-facing switch results/14 recommended.

**FireProt is not a benchmark AFToolkit publishes**, so its number has to be produced.
It also turns out to be largely inside AFToolkit's training set (below), which bounds
what the comparison can say.

## Leakage audit (the point of the exercise)

| test corpus | vs | seen in training | source |
|---|---|---|---|
| S669 (62 proteins) | this project's Tsuboyama corpus | **0** at 25 % and 30 % MMseqs2 identity | results/09 map |
| S669 (94 proteins) | AFToolkit's cDNA+PROSTATA | filtered by the authors at BLAST >36 % identity, e<0.05 | AFToolkit paper |
| FireProt ≤500 (138) | this project's Tsuboyama corpus | **8** at 30 % → scored on 130 | results/08 map |
| FireProt ≤500 (130) | AFToolkit's cDNA+PROSTATA | **90** proteins, by PDB identity | AFToolkit's released training manifest |

So both methods are genuinely blind on S669, and the S669 threshold asymmetry favours
AFToolkit's opponent (25/30 % is stricter than 36 %). On FireProt only **40 proteins /
1,265 variants** are blind to both; this project scores ρ 0.635 / r 0.583 there (vs
0.657 / 0.645 on all 130), i.e. the restriction costs it little.

The 90-protein figure comes from AFToolkit's own `cdna+PROSTATA_mut_idxs.csv`: its
2,375 PROSTATA training rows sit on 172 PDB entries, and FireProt is drawn from the same
ProTherm/VariBench lineage. 319 (protein, mutation) pairs are literally identical.
**Zero** overlap with the cDNA/Megascale half.

## Next steps

- [ ] Wait for job **20245** (`squeue -j 20245`); ~2,983 `.npy` files land in
      `/grupos/Marce/estructural/ddG_with_Boltz/aftoolkit/features`. Requeue any failed
      shard with `sbatch --array=<i>%2 aft_array.sbatch 32` — it resumes.
- [ ] `rsync -a cranex:/grupos/.../aftoolkit/features/ $AFT/fireprot_features/` then
      `AFT=... python results/16_aftoolkit_headtohead/score_fireprot.py`. It reports
      leaked and blind subsets separately; the **blind 40 proteins / 1,265 variants** is
      the comparison that means anything.
- [ ] Extend `make_figures.py` with the FireProt panel once those numbers exist.
- [ ] `build_report.py` → `report.pdf` once FireProt lands.
- [ ] Update `results/README.md`, `results/history.md` and `theory/sota_2026.md`
      (its S669 table still carries the superseded regime A/B/D numbers).

## Blockers

None open. Resolved during setup:
- AFToolkit's README download URLs are wrong: the S669 features are `s669_pkls.zip`
  (README says `s699_`), and the model paths 403 unless the `+` in
  `pair+lddt_logits+plddt` is percent-encoded as `%2B`.
- The pickles import `AFToolKit` (capital K) while the repo directory is `AFToolkit`;
  and `AFToolKit/processing/__init__.py` pulls in torch. Scoring uses a shim package
  with an empty `processing/__init__.py`. Adapters need scikit-learn 1.4.x.
- On the cluster, `openfold/utils/script_utils.py` imported the Amber relaxation path
  (openmm/pdbfixer) at module scope; patched to a lazy import inside `relax_protein`,
  which is never called. No model code touched.

## Shareable write-up

<https://claude.ai/code/artifact/d7f6d51c-b727-49f1-ab99-a131089f46d0> — the S669 result and
the leakage audit as a page (private until shared). Regenerate/update it from
`README.md` + `figures/`; keep the same URL.

## Log — newest first

### 2026-08-28 (evening) — FireProt lands; the leakage reversal replicates the S669 story

- **Extraction finished**: job 20245, 32/32 shards COMPLETED, ~13 min each, no shard
  failures. **2,899 of 2,983 variants** produced features; **84 (2.8 %) failed inside
  AFToolkit's own code** — 33 `AssertionError`, 28 `RuntimeError('No active exception to
  reraise')` (a bare `raise` in their error path), 23 `KeyError(<resnum>)`. All are
  structures with unresolved residues, where their contiguous-numbering assumption in
  `set_observable_positions` breaks. Both methods scored on the same 2,899.
- **Result** (Spearman ρ): on the 88 proteins AFToolkit trained on, AFToolkit 0.755 vs
  ours 0.716 (paired Δ −0.063 [−0.164, −0.004], significant). On the 36 blind to both,
  AFToolkit 0.633 vs ours 0.685 (Δ +0.046 [−0.019, +0.094], n.s.). **AFToolkit loses
  0.122 ρ when its own training proteins are removed; this project loses 0.031.**
- The naive all-corpus number (AFToolkit 0.706 vs ours 0.696) is the one anyone would
  compute from the two papers, and it gets the *direction* wrong.
- Cluster housekeeping while this ran: `conda clean` + `pip cache purge` took
  `/home/hgarbarino` 48 GB → 17 GB; deleting `mave_hoie_le200/{_predict_shards,slim}`
  (orphaned temp + a slim store verified byte-identical to the local copy by a
  filename+size manifest hash) took the project dir 31 GB → 23 GB. Guarded by a
  re-check of the manifest hash and a 2-hour write-freshness test immediately before
  `rm`. `scan_GLA_human` was explicitly excluded — its `_predict_shards` looked like the
  same orphaned temp but belongs to the live job 20156.

### 2026-08-28 — S669 head-to-head done (paired); FireProt extraction set up

- **S669.** Reproduced AFToolkit's published S669 result exactly from its released
  assets, then compared paired on the 541 variants this project covers. Numbers above.
  A control settles the subset worry that results/14 could only bound indirectly:
  AFToolkit scores **ρ 0.552 on the 128 variants our 500-residue cap excludes** vs
  **0.511 on the 541 we keep** — the cap gives us the *harder* half, so the comparison
  is conservative.
- **Leakage.** Re-derived both filters from the committed cluster maps. Fixed a bug in
  the first pass: `cluster_map_30.csv` stores Tsuboyama ids with a `.pdb` suffix and
  FireProt ids without, so stripping it from both matched zero Tsuboyama proteins and
  silently reported 0 leaky. Corrected → the known 8/138.
- **AFToolkit training overlap.** A first pass reported 47/130 FireProt proteins;
  that undercounted because FireProt's `pdb_id` is sometimes a pipe-separated list
  (`1AKK|1I5T`) and was compared unsplit. Correct figure: **90/130**.
- **Cluster setup** (`/grupos/Marce/estructural/ddG_with_Boltz/aftoolkit`): AFToolkit
  cloned, venv with `--system-site-packages` on the `ddG_with_Boltz` python (reuses
  torch 2.6.0+cu124, so the env is 37 MB), `params_model_2_ptm.npz` fetched from the
  AlphaFold colab tar (other four models deleted; /grupos is at 100 %, 328 GB free).
- **Reproduction validated on GPU** (job 20239, nodo6, RTX 2080 8 GB): re-deriving AF2
  features from raw PDB for `1a7v_A_A66H` and `1a0f_A_S11A` matches AFToolkit's released
  feature vectors at r = 0.99996 / 0.99994 (not bit-identical — torch 2.6 vs 1.13,
  different GPU, `low_prec=True`). Job 20240 checks whether that propagates to the
  predicted ΔΔG.
- **FireProt inputs built**: 130 PDBs fetched from RCSB; FireProt sequence numbering
  mapped to PDB residue numbering by searching every chain × constant offset and keeping
  the best (FireProt's own `chain` annotation is wrong for 6 proteins). **2,983 of 3,102
  variants** land on a PDB residue carrying the right wild-type amino acid, on 128
  proteins; the other 119 are dropped and both methods will be scored on the same set.
- **Prediction-level validation** (job 20240, 26 S669 variants, nodo6): AFToolkit's own
  SVM on our re-derived features vs on its released features — **r 0.968 / ρ 0.975,
  mean |Δ| 0.080 kcal/mol**; against experiment ρ 0.543 (ours) vs 0.558 (theirs).
  So the numerical drift costs AFToolkit roughly **−0.015 ρ**: the FireProt number we
  produce for it is a slight *under*estimate, and must be reported as such.
- **Cost estimate**: fitted t ≈ L^1.78 from measured points (L=96 → 4.7 s, L=229 → 22 s)
  ⇒ ~27 GPU-hours for 2,983 variants (2 passes each, 4 recycles), max 82 s/variant.
- **Sharding by protein failed and was abandoned** (job 20242, cancelled after 14
  variants; its features are kept — the runner skips existing `.npy`). One protein,
  **P06654 (448 aa, 860 of the 2,983 variants), is 17.4 h — 64 % of the whole cost**, so
  no by-protein split can balance and shard 5 of 16 would have blown the wall clock.
  Replaced with a longest-processing-time bin-packing over *variants*, precomputed into
  a `shard` column of `fireprot_aft_task.csv`: **32 shards of 0.85 h each** (93 variants,
  spread of 0.00 h).
- **Running now: job 20245, `--array=0-31%2`** on nodo6 + nodo8 (RTX 2080, 8 GB — no OOM
  so far; the longest chain is 477 aa). First 51 variants took ~8.8 s each against the
  32.8 s/variant the cost model assumed, so the wall clock may land nearer 4-6 h than the
  ~13.6 h estimated — the model was fitted on two points and over-predicts. `%2` deliberately,
  to leave GPUs for the user's own `ddg-predict` array (job 20156) on the same partition
  — raise to `%3`/`%4` if that array finishes first. Shards are resumable (they skip any
  variant whose `.npy` exists), so a requeue costs at most one shard.
- **Not yet run:** a larger S669 reproduction-penalty calibration (the −0.015 ρ above
  rests on 26 variants). Cheap; worth adding before quoting the FireProt gap.
