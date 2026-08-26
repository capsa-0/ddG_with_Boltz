# 15 — MAVE stability→function transfer (RF4Mave)

**Status:** 🚧 in progress. Harness validated; Boltz run on the cluster. No ΔΔG
predictions yet — see [`status.md`](status.md).

## What

Does our Boltz-embedding ΔΔG carry stability information competitive with **Rosetta
ΔΔG** for predicting **experimental MAVE fitness**? We reproduce Høie et al. 2022
(*Cell Reports* 38:110207) and swap our ΔΔG in for Rosetta's, everything else held
fixed.

## Why

Every previous benchmark in this series — Tsuboyama, FireProt, S669, Ssym — shares
ΔΔG as the target, so they all inherit the same assay conventions and curation
lineage. Results 11–13 showed the remaining error is domain shift the features cannot
express, not a fixable calibration term. This is the first test of whether the
representation carries stability information that survives a **change of question**:
the label here is cellular fitness, measured in other labs, by other assays, in units
that are not kcal/mol.

The point is *not* ΔΔG accuracy — that is Tsuboyama's job. A perfect ΔΔG predictor
caps out well below ρ = 1 here, because fitness ≠ stability. Low ρ is only meaningful
**relative to Rosetta's**.

## How

**Corpus.** The 11 Høie proteins ≤200 aa ("Tier 1"), carrying 13 of their 39 MAVE
datasets. Full L×19 saturation of each = **25,224 Boltz structures**. Full saturation
rather than measured-variants-only because their best model needs all 20 substitutions
at a position, and it costs 3.8 % more.

The ≤200 aa cap is a compute budget, and it does not tilt the comparison: median
|ρ(Rosetta ΔΔG, s_exp)| is **0.301** on these 13 datasets and **0.301** on all 39.

**ΔΔG prediction.** Regimes A (Tsuboyama), B (FireProt), D (fine-tuned) — the concat
`wtz|mtz` representation, antisymmetry augmentation, 5-seed MLP ensemble adopted in
results/07 and used by 08/09.

**Two comparison layers**, both on identical variants:
- *Direct* (their Fig 2A): per-dataset Spearman of each predictor against s_exp, no model.
- *RF4Mave* (their Fig 2B): their leave-one-protein-out random forest, run once with
  Rosetta's ΔΔG and once with ours. The **paired difference** is the result.

**Leakage.** UBI4 (ubiquitin) is the only Tier-1 protein homologous to Tsuboyama at
25 %/30 % identity — it clusters with `1UBQ.pdb`. 2 of 13 datasets. Everything is
reported full and UBI4-dropped. SUMO1 and UBE2I are clean: the ubiquitin *fold*
similarity falls below 25 % sequence identity.

## Headline numbers

**Phase 0 — harness reproduction** (their `preprocessed.pkl`, all 39 datasets, LOPO
median Spearman). The gate this experiment rests on:

| model | ours | published |
|---|---|---|
| null (s̃_exp) | 0.334 | 0.17 † |
| ΔΔG only (Rosetta) | **0.249** | 0.25 |
| ΔΔE only (GEMME) | **0.409** | 0.42 |
| ΔΔG + ΔΔE | **0.466** | 0.47 |
| position-context (47 features) | **0.519** | 0.52 |

† The null is the one model their `train.sh` does not pin with an explicit feature
regex, so its definition is inferred on our side. Our 0.334 matches their own Table S1
"MAVE WT→Mut" column (median ≈ 0.33). Reported as an open discrepancy; it does not
enter the ΔΔG comparison.

All four baselines pinned by explicit feature regexes in their `train.sh` reproduce
within **±0.011**, and the position-context set comes out at exactly **47 features** —
the count the paper states.

**Main result:** pending the Boltz run.

## Data & provenance

| what | where |
|---|---|
| Paper | `theory/biblio/marce/RF4Mave.pdf` — doi:10.1016/j.celrep.2021.110207 |
| Source data | Zenodo `10.5281/zenodo.5647207`; `github.com/KULL-Centre/papers/tree/main/2021/ML-variants-Hoie-et-al` |
| Fetched to | `data/raw/mave_hoie/` (gitignored, 252 MB) — `fetch_hoie.py` |
| Corpus | `data/raw/mave_hoie_le200.csv` (25,213 mutations / 11 proteins) |
| Labels | `data/raw/mave_hoie_le200_labels.csv` (24,499 rows: s_exp, Rosetta ΔΔG, GEMME ΔΔE, ss, rsa) |
| Config | `experiment_configs/mave_hoie_le200.yaml` |
| Processed | `data/processed/mave_hoie_le200/` (cluster; slim store synced back here) |
| Features | `data/processed/mave_hoie_le200/features_summary.parquet` |
| ΔΔG predictions | `data/processed/mave_hoie_le200/mave_ddg_predictions.csv` |
| Training corpora | `data/processed/{tsuboyama_bench_fast,fireprot_le500}/features_ablation.parquet` (**local only** — the cluster has neither) |
| Leakage map | `homology/mave_le200_leakage.csv` |
| Phase-0 tables | `phase0/lopo_summary.csv`, `phase0/lopo_per_dataset.csv` |

## Reproduce

```bash
python results/15_mave_stability_transfer/fetch_hoie.py
python results/15_mave_stability_transfer/build_corpus.py
MMSEQS_BIN=/path/to/mmseqs python results/15_mave_stability_transfer/build_homology_map.py
python results/15_mave_stability_transfer/rf4mave.py          # Phase 0 gate (~1 h)

# on cranex, after landing the config + corpus CSV:
./slurm/submit_scan.sh experiment_configs/mave_hoie_le200.yaml 256 3
# then rsync data/processed/mave_hoie_le200/{slim,features_summary.parquet,...} back

python results/15_mave_stability_transfer/check_frames.py     # verify the feature rebuild
python results/15_mave_stability_transfer/predict_ddg.py      # regimes A/B/D
python results/15_mave_stability_transfer/score.py            # both comparison layers
```

## Notes for whoever picks this up

- The embeddings are kept (`slim.keep_s: true`, ~5.8 GB) precisely so **other models
  and feature blocks can be tried without re-running Boltz**. The slim store keeps the
  mutated residue's full `z` row, the matching `pdistogram` row, and full `s` — lossless
  for anything derived from those. It does *not* keep the rest of the L×L pair tensor,
  so a feature needing couplings between arbitrary residue pairs would need a re-run.
- `head.mode: inference` is deliberate. MAVE fitness is not ΔΔG and must never occupy
  the `ddg` column; it is joined back on `(uniprot, mutation)` after the fact.
- `rf4mave.py` follows their released **code**, not the paper prose — the two differ on
  details that matter (the −100 missing sentinel, the `-x 2` filter, the exact RF
  hyperparameters). See the module docstring.
