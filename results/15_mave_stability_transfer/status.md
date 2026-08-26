# Status — 15_mave_stability_transfer

**State:** 🚧 In progress — Phase 0 (harness reproduction) running; corpus + leakage map done. No GPU submitted yet.
**Last updated:** 2026-08-26

## Current state

Testing whether our Boltz-embedding ΔΔG carries stability signal competitive with
Rosetta ΔΔG for predicting **MAVE functional fitness** (Høie et al. 2022, Cell Reports
38:110207 — `theory/biblio/marce/RF4Mave.pdf`). Planned in `TODO.md` §4.

Done so far, all CPU-only on the workstation:
- Høie data fetched and verified usable (39 datasets / 29 proteins / 212,450 scored
  single variants; 100 % WT-residue agreement with their header sequences).
- Tier-1 corpus built: **11 proteins ≤200 aa, 13 MAVE datasets, 25,224 Boltz
  structures**, 0 WT mismatches.
- Homology leakage map built: **only UBI4 (ubiquitin) is leaky** vs Tsuboyama.
- Phase-0 harness reproduction of their published LOPO medians: in progress.

## Next steps
- [ ] Finish Phase 0; check the four RF baselines against 0.25 / 0.42 / 0.47 / 0.52.
- [ ] Push, `git pull` on cranex, submit
      `./slurm/submit_scan.sh experiment_configs/mave_hoie_le200.yaml 256 3`.
- [ ] After ~5 shards, measure real s/structure and re-derive the ETA before letting
      the rest run (the failure mode that killed the results/10 full scan).
- [ ] `rsync` the slim store (~4.1 GB, `keep_s: true`) back here so different models
      and feature blocks can be tried without the cluster.
- [ ] Phase 3: `predict_ddg.py` (regimes A/B/D) → `score.py` (direct + LOPO layers).

## Blockers
- None.

## Log — newest first

### 2026-08-26 — data verified, corpus + leakage map built, Phase 0 running

**Data availability confirmed.** `data.zip` (63 MB) from
`github.com/KULL-Centre/papers/tree/main/2021/ML-variants-Hoie-et-al` (also Zenodo
`10.5281/zenodo.5647207`). `fetch_hoie.py` pulls the 39 merged PRISM tables, their
`preprocessed.pkl` (the built 47-feature tables) and `mut_matrix_alphabetical.npy`
into `data/raw/mave_hoie/` (gitignored, 252 MB).

Checked all 39 datasets: **100 % of variants satisfy `sequence[pos-1] == wt_aa`**
against the header sequence, so they pass `ddg.datasets.prepare`'s validation
untouched. Only 3 of the 39 are true VAMP-seq abundance assays (PTEN 003, TPMT 014,
NUDT15 005); `012_P53_abundance_reversed` is misleadingly named — its header says
growth/phenotype (Giacomelli 2018).

Recomputed the direct per-dataset Spearman baselines from their columns; they
reproduce Table S1. Median |ρ(Rosetta, s_exp)| = **0.301** over all 39.

**Scope decision (user): Tier 1 = the 11 proteins ≤200 aa.** 13 MAVE datasets.
Median |ρ(Rosetta, s_exp)| on this subset is also **0.301**, so the size cap does not
favour or disfavour the stability baseline. Full L×19 saturation (not just measured
variants) because the position-context model needs all 20 substitutions at a
position and it costs only 3.8 % more.

| protein | L | datasets | mutations | structures | scored rows |
|---|---|---|---|---|---|
| CALM1 | 149 | 1 | 2831 | 2832 | 1813 |
| GAL4 | 65 | 1 | 1235 | 1236 | 1196 |
| GmR | 177 | 1 | 3363 | 3364 | 1929 |
| HRas | 189 | 1 | 3591 | 3592 | 3135 |
| IF-1 | 72 | 1 | 1368 | 1369 | 1368 |
| NUDT15 | 164 | 2 | 3116 | 3117 | 5856 |
| PAB1 | 75 | 1 | 1425 | 1426 | 1188 |
| SUMO1 | 101 | 1 | 1919 | 1920 | 1700 |
| UBE2I | 159 | 1 | 3021 | 3022 | 2563 |
| UBI4 | 75 | 2 | 1425 | 1426 | 2575 |
| ccdB | 101 | 1 | 1919 | 1920 | 1176 |
| **total** | | **13** | **25,213** | **25,224** | **24,499** |

`build_corpus.py` → `data/raw/mave_hoie_le200.csv` (corpus) +
`data/raw/mave_hoie_le200_labels.csv` (24,499 rows; 23,444 with Rosetta ΔΔG, 24,445
with GEMME ΔΔE). WT-identity check asserted at build time: **0 mismatches**.

Config: `experiment_configs/mave_hoie_le200.yaml`, derived from `scan_GLA_human.yaml`.
Two deliberate differences: `head.mode: inference` (the label is fitness, not ΔΔG, and
must never occupy the `ddg` column — it is joined back afterwards on
`(uniprot, mutation)`), and **`slim.keep_s: true`** (the scan template ships `false`;
`s` is the only retained field the concat model does not read, but it is what `sdim_*`
features need, and dropping it would mean re-running Boltz. Cost 1.4 → 4.1 GB).

**Leakage map** (`build_homology_map.py`, MMseqs2 15-6f452 at 25 %/30 % id, 80 % cov,
561 pooled sequences = 412 Tsuboyama + 138 FireProt + 11 MAVE):
**UBI4 is the only leaky protein**, at both thresholds, clustering with Tsuboyama's
`1UBQ.pdb`/`1SIF.pdb`/`2MLB.pdb` — i.e. ubiquitin itself. 2,575 of 24,499 scored rows
(10.5 %), 2 of 13 datasets. SUMO1 and UBE2I are **clean** — the ubiquitin *fold*
similarity is below 25 % sequence identity. → report full and UBI4-filtered numbers.
`mmseqs` is on neither this workstation nor cranex; used a static binary via
`MMSEQS_BIN`, same as results/09.

**Phase 0 (harness reproduction), running.** `rf4mave.py` re-implements their LOPO
protocol, decoded from their released code rather than the paper prose:
`RandomForestRegressor(n_estimators=150, max_features="sqrt", min_samples_leaf=15)`;
missing values are the sentinel `-100`, not NaN; their `-x 2` filter drops rows whose
own Rosetta *or* GEMME value is missing, from train and validation alike; features are
selected with the same `str.contains` regexes their `train.sh` passes via `-f`; for
each of the 39 datasets every dataset of the same protein leaves training. Their 47
"position-context" features decode exactly as 20+1+1 Rosetta + 20+1+1 GEMME + 3 s̃.

First result: `null_smave` median ρ = **0.334** vs the paper's quoted 0.17 (which the
text gives as a *mean*, and which is the one baseline `train.sh` does not define via an
explicit `-f` regex). Our 0.334 agrees closely with Table S1's own direct
"MAVE WT→Mut" column (median ≈ 0.33), so the discrepancy is most likely about how the
green box in their Figure 2B was defined, not a harness bug. The four baselines that
*are* pinned by explicit regexes in `train.sh` (ΔΔG-only 0.25, ΔΔE-only 0.42, both
0.47, position-context 0.52) are the real gate — pending.
