# 14 — methods & provenance appendix

Per-number provenance behind the README's summary statements.

## Feature blocks

Every block is declared as `(invariant, wt-side, mt-side)` columns, and each block has a
**form** that fixes its antisymmetry transform:

| form | reverse-mutation transform | blocks |
|---|---|---|
| `concat` | swap the wt/mt halves, negate ΔΔG | `z`, `cw`, `far`, `onehot`, `bio*`, `cons` |
| `diff` | **negate the whole vector**, negate ΔΔG | `dz`, `diag`, `cwpool` |

`run_ablation.augment()` dispatches on form and refuses mixed configs. This matters:
the original code applied the half-swap unconditionally, which is meaningless on a
difference-form block, so **`dz` + augmentation had never been computable** before this
experiment. Mixed-form configs (`base+diag`) are valid only under `--no-augment`.

### Pair-track blocks

With `wt_row = z_wt[i,:,:]`, `mut_row = z_mut[0,:,:]` at the mutated position `i`:

| block | definition | dims | form |
|---|---|---|---|
| `zdiag` (`diag`) | `mut_row[i] − wt_row[i]` — the **diagonal**, unpooled | 128 | diff |
| `zpool` | `mean_j(mut_row[j] − wt_row[j])` — uniform pooled difference | 128 | diff |
| `wtz`/`mtz` (`z`, "concat") | `mean_j wt_row[j]`, `mean_j mut_row[j]` — pooled **levels** | 2×128 | concat |
| `wtcw`/`mtcw` (`cw`) | contact-weighted pooled levels | 2×128 | concat |
| `wtfar`/`mtfar` (`far`) | far-shell-weighted pooled levels — the negative control | 2×128 | concat |
| `cwd` (`cwpool`) | `mtcw − wtcw` — contact-weighted pooled **difference** | 128 | diff |
| `onehot` | one-hot (wt_aa, mut_aa) — substitution identity, no structure | 2×20 | concat |

Algebraically `zpool = mtz − wtz`, so concat contains the pooled information of Δz plus
the absolute levels; **the diagonal is the one thing Δz has that concat lacks.**

### Contact weights

From the **wild-type** distogram row at the mutated position (using the mutant's own
would make the weight mutation-dependent and leak the perturbation into the pooling):

```
p_ij  = softmax(pdrow_wt[i, j, :])       # 64 bins, boundaries linspace(2, 22, 63)
w_ij  = Σ p_ij over bins with lower edge < 8 Å,  masked to |i − j| > 2
w_ij /= Σ_j w_ij
```

The `|i−j| > 2` mask matters: without it the adjacent backbone dominates every weight
vector identically. `far` uses the renormalised complement `P(d ≥ 8 Å)` under the same
mask, so it is matched in dimensionality, form and construction — only the shell differs.

### Biophysics and conservation blocks

*Site (swap-invariant):* contact numbers at 8/10/12 Å, `site_cn_z` (contact number
z-scored **within the protein**), relative position, chain length, distance to the
nearer terminus. *Residue (swapped):* volume, Kyte–Doolittle hydropathy,
Fauchère–Pliška transfer free energy, charge at pH 7, polar/aromatic flags, Chou–Fasman
helix and sheet propensities, Vihinen flexibility, Gly/Pro flags. *Interactions:* each
residue scalar × `site_cn_z`. `bio_t` keeps only the dimensionless site columns.

*Conservation:* a3m columns map 1:1 onto WT residues after dropping lowercase insertions;
80 %-identity sequence weighting; pseudocount 1.0 over Robinson & Robinson background.
Depth capped at 2,000 by **random subsampling** (a3m rows are E-value ordered, so
truncation would keep only close homologues and understate entropy); `msa_depth` reports
true pre-subsampling depth. `msa_has_msa` thresholds at depth ≥ 10 (the original
`depth > 1` was constant at 1.0 — a dead feature).

## Evaluation protocol

- **Training:** `tsuboyama_bench_fast`, 12,359 mutations / 412 proteins.
- **Split:** `GroupKFold(5)` on `wt_id`; out-of-fold predictions.
- **Model:** `make_model("mlp")` — median impute → StandardScaler → 5-seed
  `VotingRegressor` of `MLPRegressor((256,128,64), alpha=3e-3, batch_size=256,
  early_stopping=True)`. `n_jobs=2` for this workstation; parallelism does not change fits.
- **Transfer:** the five fold models are averaged onto the blind corpus.
- **Augmentation** is treated as an experimental **factor**, not a fixed default.
- **Sanity gates:** `base` reproduces results/07's concat+symmetry Tsuboyama number
  (r = 0.799); `dz` reproduces results/05's FireProt transfer (0.647 vs 0.648).

### Sign conventions

Tsuboyama and FireProt both store ΔΔG positive-is-destabilizing (78 % / 68 % positive).
**S669 is inverted** (25 % positive) and is negated on load. The `TRANSFER` registry in
`run_ablation.py` carries a per-corpus `flip` flag so this cannot silently regress.

### FireProt assembly

`fireprot_le500` has a feature table but no slim store; its inputs are the two shards
that do. Verified: `fireprot_le200` (1,543 / 85) ∪ `fireprot_201to500` (1,662 / 53) =
3,205 unique `(wt_id, mutation)` pairs covering 3,205/3,205 of `fireprot_le500`'s rows.

### S669 duplicate variants

S669 contains 17 variants twice (same `wt_id`+`mutation`, different measured ΔΔG —
repeat measurements). Their feature vectors are byte-identical in both tables (verified
with `np.allclose`), so the new tables are deduplicated and joined **many-to-one**,
keeping both measurements as separate labelled rows sharing one feature vector.

## Homology filtering

Using the joint MMseqs2 cluster map built in results/08 (`splits/cluster_map_30.csv`,
412 Tsuboyama + 138 FireProt WT sequences, 80 % coverage):

| threshold | FireProt proteins sharing a Tsuboyama cluster | variants |
|---|---|---|
| 30 % | 8 / 138 | 103 / 3,205 (3.2 %) |
| 50 % | 8 / 138 | 103 (3.2 %) |
| 90 % | 5 / 138 | 59 (1.8 %) |

The 8 proteins: P03040, P0A9X9, P13123, P19614, P32081, P39476, P41016, P61991.
`wt_id` overlap is **zero**, which is all results/05 checked — an identifier test, not a
homology test. Filtered prediction dumps are written as
`data/processed/_analysis/exp14_fpfilt_*.csv` (130 proteins) and are the **primary**
basis for every FireProt claim; unfiltered numbers are reported alongside and differ
negligibly. **S669 requires no filter**: results/09 established zero Tsuboyama
homologues at 25 % identity.

## Cluster bootstrap

`bootstrap.py`: resample **proteins** with replacement (the unit of independence —
mutations within a protein share a structure, an embedding and an assay batch), take all
their mutations, recompute each metric, 400 times. Reported as the **paired difference
against a reference configuration on the same resample**, so the shared draw cancels.
Resamples with fewer than 10 stabilizing variants are skipped; all runs used 400/400.
Operates on saved predictions — no refitting. Output files carry the reference config in
the filename (`bootstrap_<dump>_ref-<cfg>.csv`).

Why S669 resolves less: 62 protein clusters, of which **5 proteins hold 51 %** of the
541 variants (median protein contributes 3). Leave-one-protein-out moves the `cw`−`base`
r gap across [+0.053, +0.122] around a full-sample +0.082.

## Metrics

Definitions follow `results/13_balanced_loss/run_balanced.py`, `STAB = −0.5` kcal/mol.

**`detpr30` is retired.** Computed from the 30 most-stabilizing predictions, it returned
+0.099, +0.076 and **−0.194** across the three evaluation sets — opposite signs, never
significant. It remains in the tables for continuity with results/13; no claim uses it.

## Figure palette

`#1F6FB4 / #D95F02 / #1B9E77 / #7570B3`, selected by running the colour-vision validator
rather than by eye: all six checks pass (worst adjacent pair ΔE 11.6 deuteranopia, 19.9
normal vision). The palette used elsewhere in `results/` (`#4F5D5A` with `#0E6C68`)
**fails** at ΔE 2.3 deutan / 6.4 normal — effectively one colour for those readers.
