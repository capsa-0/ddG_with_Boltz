# Status — 15_mave_stability_transfer

**State:** ✅ Done. Full corpus (25,224 structures, 0 gaps), both comparison layers scored, bootstrap CIs in, figures rendered.
**Last updated:** 2026-08-27

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
- [ ] **The MSA-confound test** (the one open scientific question): rebuild this corpus
      with `no_msa: true` and re-run. Separates "our ΔΔG is a better stability term"
      from "our ΔΔG smuggles in conservation Rosetta cannot have".
- [ ] Optional Tier 2 (≤250 aa) to add TPMT + HSP82 and tighten a CI whose lower
      bound is +0.008.
- [x] ~~After array 1415 finishes: backfill the 6 lost shards~~ (98 + 132–136,
      ~594 structures, 2.4 % of the corpus) before trusting the feature table.
      The cluster checkout is on `main` and does **not** yet have the nodo4 exclude,
      so land the two updated files first, then re-submit:
      ```bash
      B=origin/results/11-12-calibration-and-error-anatomy
      git fetch origin results/11-12-calibration-and-error-anatomy
      git show $B:slurm/predict_array.sbatch > slurm/predict_array.sbatch
      git show $B:slurm/submit_scan.sh      > slurm/submit_scan.sh
      ./slurm/submit_scan.sh experiment_configs/mave_hoie_le200.yaml 256 3
      ```
      (These two are *tracked* on the cluster, unlike the config/CSV landed earlier,
      so this leaves them showing as modified until the branch is merged.)
      predict skips everything already in the slim store, so it redoes only the gap.
      Verify afterwards: slim structure count should reach 25,224.
- [ ] After ~5 predict shards land, measure real s/structure and re-derive the ETA
      before letting the rest run (the failure mode that killed the results/10 full
      scan). Budget is extrapolated from 65 s/structure at 398 aa; the exponent for a
      65–189 aa protein is unverified.
- [x] `check_frames.py` — score.py's feature rebuild verified (max |Δ| 0.042, PASS).
- [ ] `rsync` the slim store (~4.1 GB, `keep_s: true`) back here so different models
      and feature blocks can be tried without the cluster.
- [ ] Phase 3: `predict_ddg.py` (regimes A/B/D) → `score.py` (direct + LOPO layers).

## Blockers
- None.

## Log — newest first

### 2026-08-27 — reproducimos su Figura 1 con nuestro ΔΔG; y una primera cota al confundidor del MSA

**Qué se hizo.** `paper_figures.py` (nuevo, local, sin GPU, ~2 min): reproduce la Figura 1
de Høie et al. poniendo nuestro ΔΔG en el lugar del de Rosetta, y agrega un análisis que
las correlaciones del `score.py` no pueden dar — de dónde viene nuestra ventaja.

**Gate de fidelidad, primero.** Con el ΔΔG *de ellos* sobre nuestro subconjunto Tier-1 de
13 datasets, las dos esquinas que el paper cita en el texto vuelven a salir:

| sector | nosotros | ellos publican (39 datasets) |
|---|---|---|
| ΔΔE < 0,25 y ΔΔG < 2,0 → % alta fitness | **84 %** | 81 % |
| ΔΔE > 0,75 y ΔΔG > 4,5 → % baja fitness | **96 %** | 93 % |

`_check_orientation` aborta si estos se van a más de 15 puntos, así que las figuras no se
generan sobre un arnés roto.

**Gotcha de datos, verificado y no asumido.** La columna `gemme_dde` de
`data/raw/mave_hoie_le200_labels.csv` es el `gemme_score_01` de PRISM tal cual, y corre
**al revés de la ΔΔE del paper**: alto = evolutivamente tolerada. Correlaciona
*positivamente* con fitness (ρ agrupado +0,27) — por eso `layer1_direct.csv` muestra
`rho_gemme > 0` mientras los dos brazos de ΔΔG son negativos. Las figuras usan
ΔΔE = 1 − `gemme_dde`, y esa elección se valida contra los porcentajes publicados, no
contra el nombre de la columna. Documentado en `figures/README.md`.

**Resultado 1 — el paisaje se reproduce, pero sobre un eje comprimido.** sd de nuestro
ΔΔG 0,97 kcal/mol contra 2,14 de Rosetta, así que con los cortes absolutos del paper
(2/3/4,5) la columna alta queda casi vacía (46 variantes contra 1.129 de Rosetta). Con
cortes **al mismo cuantil** los sectores se llenan y la estructura es prácticamente
idéntica a la de Rosetta: fila de ΔΔE alta 68/81/88/96 % contra 71/81/88/96 %.
La compresión de amplitud ya era conocida (results/05, pendiente 0,27) — esto la vuelve
visible en el marco del paper.

**Resultado 2 (nuevo) — más de la mitad de nuestra ventaja sobre Rosetta se explica por
conservación.** AUC para detectar pérdida de función, bootstrap de clusters sobre las 11
proteínas, pareado:

| | Rosetta | nuestro | Δ | IC 95 % |
|---|---|---|---|---|
| agrupado | 0,710 | 0,759 | **+0,048** | [+0,014, +0,079] ✔ |
| dentro de cada cuartil de ΔΔE | 0,633 | 0,654 | **+0,021** | [−0,010, +0,052] ✗ |

Condicionar sobre conservación se lleva el 56 % de la ventaja, y lo que queda ya no
despega de cero. **Es la primera cota cuantitativa al confundidor del MSA que el README
deja abierto, y se obtuvo sin GPU.** Lectura honesta en las dos direcciones: el residuo es
positivo en los cuatro estratos (+0,018 a +0,023), así que "no despega de cero" es falta de
potencia con 11 proteínas, no ausencia de efecto. **No reemplaza el test `no_msa: true`** —
lo hace más urgente, y le da una predicción: si la ganancia fuera puramente evolutiva, el
condicional debería caer a ~0.

El estadístico condicional se calcula como **una cantidad por resample** (media de los
cuatro AUC de estrato), no promediando los cuatro intervalos — promediar IC no da un IC.

**Salidas:** `figures/03_landscape_reproduction.png`, `figures/04_conservation_strata.png`,
`conservation_strata_auc.csv`, `paper_figures.py`. Nada se re-entrenó: usa
`mave_ddg_predictions.csv` (regime `mean`) ya en disco.

**Pendiente si esto se promueve:** las tres figuras que faltan de su Fig 2A (movimiento por
dataset) ya están cubiertas por `02_per_dataset_direct.png`; el README del folder todavía no
cita `03`/`04`.

**Addendum — el set de VAMP-seq, mirado solo.** `005_NUDT15_abundance` es el **único**
VAMP-seq del corpus Tier-1 (PTEN y TPMT, los otros dos del paper, quedaron fuera por el
tope de 200 aa). Es nuestro mejor dataset de los 13 en las dos capas. |rho| directo sobre
las 2.801 filas con cobertura emparejada, bootstrap de clusters sobre las **156 posiciones**
(dentro de una proteína la posición es la unidad: las 19 sustituciones de un sitio comparten
entorno, enterramiento y columna del MSA), 4.000 remuestreos:

| contraste | Δ\|rho\| | IC 95 % |
|---|---|---|
| nuestro (0,675) − Rosetta (0,531) | **+0,144** | [+0,075, +0,213] ✔ |
| nuestro (0,675) − GEMME (0,333) | **+0,340** | [+0,251, +0,433] ✔ |

**La disociación doble, dentro de una sola proteína.** NUDT15 aporta dos datasets — abundancia
(VAMP-seq, lee estabilidad) y sensibilidad a droga (lee función). Misma secuencia, misma
estructura, mismo MSA, **las mismas predicciones nuestras**; lo único que cambia es el ensayo:

| \|rho\| directo | abundancia (VAMP-seq) | sensibilidad a droga |
|---|---|---|
| GEMME (conservación) | 0,333 | **0,554** |
| Rosetta (ΔΔG) | 0,531 | 0,270 |
| nosotros (ΔΔG) | **0,675** | 0,443 |

El ordenamiento se da vuelta entero. Es la tesis central del paper reproducida dentro de una
proteína, y es la mejor evidencia de que nuestro número mide **estabilidad** y no "ser un buen
predictor de efecto de variante" en general.

**Contra-evidencia parcial al confundidor del MSA.** En este set GEMME es *débil* (0,333) y
nosotros sacamos el doble. Si nuestra ganancia fuera mayormente conservación contrabandeada,
deberíamos andar mal justo donde la conservación anda mal — y pasa lo contrario. No anula el
análisis por estratos de arriba (podemos ser parte-conservación y parte-estructura, que es lo
que ambos resultados juntos sugieren), pero acota la hipótesis: **no somos solo conservación.**

**Curiosidad con consecuencia práctica:** en este dataset agregar GEMME al modelo *empeora* el
nuestro, 0,518 → 0,429 (ddg_only → ddg_dde), y el position-context llega a 0,439. O sea, sobre
el ensayo de estabilidad más puro del corpus, **nuestro modelo de ΔΔG solo le gana a todos los
modelos más ricos**. Con Rosetta no pasa (0,427 → 0,446). Consistente con que el RF se
distraiga con una feature que no sirve para este ensayo.

Números en `layer1_direct.csv` y `layer2_lopo_per_dataset.csv`; el bootstrap por posiciones fue
ad hoc en esta sesión y **no está guardado como script** — si se promueve, hay que fijarlo.

### 2026-08-27 — FINAL: corpus complete, result stands, one confound left open

Backfill 1809 (shards 4–8 of 16 = old 133–136) + features 1810 completed. Store is
**25,224 / 25,224 structures, deficit 0**; features table **25,213 rows × 899 columns**;
`dataset_report.json` shows 25,213 in, 25,213 out, **0 drops of any kind**.

**Final numbers moved almost nothing from the preliminary 24,620-row run** (ΔΔG-only Δ
+0.071 → +0.075, every figure within 0.004), so the 593 missing structures were not a
biased slice. Worth having verified rather than assumed.

**Layer 1 — direct, no model** (median signed ρ): Rosetta −0.301 / −0.301 UBI4-dropped;
**Boltz −0.373 / −0.417**; GEMME +0.497 / +0.500.

**Layer 2 — LOPO, with 95 % protein-bootstrap CI on the paired difference:**

| feature set | Rosetta | Boltz | Δ | 95 % CI | verdict |
|---|---|---|---|---|---|
| null (s̃) | 0.352 | — | — | — | — |
| ΔΔE only | 0.430 | — | — | — | — |
| **ΔΔG only** | 0.279 | **0.354** | **+0.075** | [+0.008, +0.117] | **excludes 0** |
| ΔΔG + ΔΔE | 0.469 | 0.470 | +0.000 | [−0.036, +0.038] | null |
| position-context | 0.510 | 0.503 | −0.007 | [−0.011, +0.007] | null |

UBI4-dropped is the same story: ΔΔG-only Δ +0.075 [+0.007, +0.123]; the other two span
zero. We are *worse* than Rosetta on both UBI4 datasets — the only two of thirteen we
lose — which is the opposite of what leakage would produce.

**Correction to an earlier entry.** On the preliminary (row-short) data the
position-context arm looked *significantly* worse (Δ −0.007, CI [−0.023, −0.002]).
On the complete data that CI is [−0.011, +0.007] and spans zero. There is **no**
position-context deficit; it is a null. The preliminary interval was an artifact of the
585 rows missing from the Boltz arm only.

**The open confound.** The pattern — a clear ΔΔG-only gain that vanishes once GEMME is
present — is what you would expect if our ΔΔG is partly an *evolutionary* predictor.
Boltz's trunk is conditioned on the MSA; Rosetta's calculation is not. results/04 put
the MSA's worth to this model at ~0.08–0.10 r, close to the +0.075 gap. That is a
hypothesis and is written up as one. The `no_msa` config from results/04 tests it
directly and is the single most valuable follow-up.

**Figures.** `01_lopo_paired.png` (bars + forest of the paired difference),
`02_per_dataset_direct.png` (dumbbell, 13 datasets × 3 predictors). Palette validated
with the dataviz six-checks at `--pairs all` (worst CVD ΔE 9.2, normal-vision 24.0).
Both re-rendered once after inspection to fix a legend/marker collision, a
mid-word-truncated axis label, and a forest panel floating in empty space.

**Ops note.** Two background jobs (the first final-analysis run and the 5.8 GB slim
sync) were killed simultaneously mid-run with no OOM trace — a session-level interrupt.
Restarted **sequentially** rather than concurrently: fitting 15 MLPs alongside a 5.8 GB
rsync on a 6.8 GB box was poor sequencing regardless of what stopped them.



### 2026-08-27 — PRELIMINARY RESULT (24,620 of 25,213 rows) + a shard-collision scare

Chain 1414→1417 finished: slim sweep 1416 was an 8 s no-op (every shard had
self-slimmed; 0 raw leftovers), features 1417 took 21 min → a 105 MB parquet with
**899 columns** = 512 concat/Δz + **384 `sdim`** + 3 keys. `slim.keep_s: true` did what
it was for: the `s` track is banked and future models can use it with no GPU.

Ran Phase 3 on the incomplete table (24,620 rows) to validate the path end to end.

**Layer 1 — direct per-dataset Spearman (median, signed):**

| predictor | full | UBI4-dropped |
|---|---|---|
| Rosetta ΔΔG | −0.305 | −0.305 |
| **Boltz ΔΔG (ours)** | **−0.375** | **−0.412** |
| GEMME ΔΔE | +0.495 | +0.498 |

**Layer 2 — leave-one-protein-out RF (median ρ over 13 datasets):**

| model | Rosetta | Boltz | Δ | Δ (UBI4-dropped) |
|---|---|---|---|---|
| null (s̃) | +0.352 | — | — | — |
| ΔΔE only | +0.430 | — | — | — |
| **ΔΔG only** | +0.279 | **+0.350** | **+0.071** | **+0.071** |
| ΔΔG + ΔΔE | +0.469 | +0.469 | 0.000 | +0.020 |
| position-context | +0.510 | +0.504 | −0.006 | −0.011 |

Note Rosetta's ΔΔG-only is **+0.279** on our 13 datasets, not the published 0.249 over
39 — which is exactly why the plan re-runs both arms through the same harness instead of
comparing against the paper's number.

**Reading it.** A substantial ΔΔG-only gain that is **fully absorbed once GEMME enters**.
The leading explanation is that **Boltz sees the MSA**: our ΔΔG carries evolutionary
signal that Rosetta's pure-physics calculation structurally cannot, and that signal is
redundant once conservation is supplied explicitly. results/04 measured MSA as worth
~0.08–0.10 r to this model — close to the +0.071 gap. Testable with the existing
`no_msa` config: if single-sequence Boltz ΔΔG still beats Rosetta on ΔΔG-only, the gain
is genuinely structural.

Per-dataset detail supports a real signal rather than leakage:
- **NUDT15 VAMP-seq abundance** (the purest stability assay): Rosetta −0.529, ours
  **−0.660** — our biggest win, where stability should matter most.
- **CALM1** (the ΔΔG-blind control): −0.079 vs −0.114, both ≈ 0. No false signal.
- **UBI4** (the one leaky protein): Rosetta −0.297/−0.440, ours −0.205/−0.346 — **we are
  worse on the leaked protein**, and the ΔΔG-only gap is identical with UBI4 dropped.

**Still preliminary, two reasons.** (1) The arms are not yet on identical rows —
Rosetta 23,415 vs Boltz 22,830; coverage matching handles Rosetta's gaps but not the 585
rows missing from ours. (2) No confidence interval yet; a median over 13 datasets moves
easily and +0.071 needs the protein-clustered bootstrap before it is defensible.

**Shard-collision scare (resolved, no data lost).** The backfill was submitted with
N=16 while the original run used N=256, so `slim --shard i/16` writes the same
`s000i.npz` as `slim --shard i/256`. `s0002.npz` jumped to 48 MB and I cancelled jobs
1800/1801 on the spot, fearing 16 shards of banked embeddings were being clobbered.
They were not: **`slim` merges into an existing shard file rather than overwriting it** —
s0002 went 99 → **198** structures (its own 99 plus the 99 recovered from old shard 98),
and neighbours stayed at 99. The cancel cost time and was not needed; it was still the
right call under uncertainty, since risking ~1,600 structures to save a delay is a bad
trade.

Useful fact that fell out: 256 = 16 × 16, so **N=256 shard *k* maps entirely into N=16
shard *k* mod 16**. The six lost shards (98, 132–136) land in N=16 shards {2, 4, 5, 6, 7,
8}; 2 is already recovered. Resubmitted as **1809** (`--array=4-8%3`, only the shards with
work) → features **1810**. Store is at 24,829 / 25,224; deficit **395**.



### 2026-08-26 — shard 1415_98 hung on nodo4; cancelled, throughput restored

At the halfway mark (128/256) the rate had dropped from 0.246 to 0.129 shards/min.
Cause: **shard `1415_98` had been RUNNING 4 h 06 m on nodo4** against a normal 13–20 min,
holding one of the three concurrent slots.

It was hung, not slow: its logs stopped at **10:23** and it was then **14:30** — four
hours with no output, progress bar frozen at structure 85/99, ~7:48 into prediction.
No traceback, no CUDA error; it simply stopped. nodo4 is not obviously a bad node — it
ran shard `1415_1` in 9 m 52 s earlier — so this reads as a one-off stall (NFS or GPU
wedge) rather than a node to add to the exclude list. Watching for a repeat.

`scancel 1415_98`. Verified afterwards: shard gone from the queue, slots refilling
(`1415_130`, `1415_131` picked up immediately), and **the chain is intact** — 1416/1417
are still plain `PENDING`, not `DependencyNeverSatisfied`, because `submit_scan.sh`
attaches the slim sweep with `afterany` precisely so one dead shard cannot strand the
run.

**Shard 98's ~99 structures are simply missing** — no raw NPZs were left behind (Boltz
appears to write its outputs at the end of a run, not per structure). Recovery is the
documented one: after the array finishes, re-run
`./slurm/submit_scan.sh experiment_configs/mave_hoie_le200.yaml 256 3`, which skips
everything already in the slim store and redoes only the gap. **Do not skip this step** —
without it the corpus is short ~99 of 25,224 structures (0.4 %), which would silently
shrink the feature table rather than fail loudly.

Revised ETA with three slots restored: ~127 shards × ~15 min ÷ 3 ≈ **10.5 h**, so around
01:00 on 2026-08-27, plus a short gap-filling run.

### 2026-08-26 — dry-run of Phase 3 on synthetic predictions; found and fixed a fairness bug

With the GPU run a third done, exercised the Phase-3 path that `check_frames.py` did
not cover, using a synthetic predictions CSV with the exact schema `predict_ddg.py`
emits (values meaningless — only the plumbing under test).

**Layer 1 join verified.** All 13 datasets join cleanly on `(wt_id, mutation)` with the
right row counts, and `rho_rosetta` comes out at median **−0.301** — exactly the direct
Rosetta baseline computed independently from the PRISM tables, per dataset as well as in
aggregate. Sign is negative as it must be (destabilizing → low fitness); the sign guard
stays quiet.

**Found: the two arms were not actually paired.** Our scan is *full saturation*, so it
has a ΔΔG for **100 %** of scored variants and **95.0 %** of position-grid cells (19/20 —
the WT cell has no mutation). Rosetta has **95.7 %** and **90.9 %**; their calculations
have genuine gaps. Two consequences, both of which would have inflated our arm:

1. The position-context model would have given the Boltz arm denser features — winning
   partly on *coverage* rather than on ΔΔG quality.
2. Worse, their `-x 2` filter drops rows whose own stability value is missing, and it
   reads whichever column is in the ΔΔG slot. So the Boltz arm would have been scored on
   ~4.3 % **more rows** than the Rosetta arm — and precisely the rows Rosetta could not
   compute, which are unlikely to be a random sample.

**Fix:** `build_frames(..., match_coverage=True)` (now the default) masks our ΔΔG to
Rosetta's availability pattern, so both arms see identical missingness and identical row
sets. Verified: both arms now report 95.7 % own-value and 90.9 % grid coverage. The
paired difference now isolates ΔΔG *quality*, which is the thing under test.
`--no-match-coverage` measures separately what the full-saturation advantage is worth —
a real benefit of our method, but one that deserves to be reported as its own number
rather than smuggled into the headline.

### 2026-08-26 — feature rebuild verified (check_frames.py): PASS

The other half of the harness. Phase 0 validated the LOPO *using Høie's own feature
tables*; Phase 3 needs `score.py` to rebuild those 47 features from the raw PRISM
tables so our ΔΔG can take Rosetta's place. That rebuild is our code, and a divergence
would corrupt the headline number in a way Phase 0 cannot see.

Same LOPO (60 trees), same 13 Tier-1 datasets, run over both feature sources:

| | median ρ |
|---|---|
| their `preprocessed.pkl` | +0.502 |
| our rebuild from PRISM | +0.510 |

**max \|Δ\| = 0.042, mean \|Δ\| = 0.019**, both feature sets exactly 47 columns.
PASS (bar was 0.05). Deviations split 8 positive / 5 negative — the largest is UBI4
dextrose (+0.042).

The residual +0.008 median offset does not affect the result: **both arms of the Phase-3
comparison go through this same rebuild** (Rosetta's ΔΔG and ours are swapped into
identical frames), so any offset relative to their pkl cancels in the paired difference.

Also fixed a benign `All-NaN slice` RuntimeWarning in `score.py` — positions with no
value for any of the 20 substitutions correctly give NaN, which becomes their -100
sentinel; the warning was noise, not a bug.

### 2026-08-26 — throughput measured: the budget was 2x pessimistic

**prepare 1414 COMPLETED in 21 min** (not the 3–5 h estimated) and built all 25,224
MSAs + queries. Small proteins make short alignments.

**First predict shard: 99 structures in 9 min 52 s** on nodo4 — ~6.0 s/structure
including ~3 min of Boltz startup, so ~4.2 s/structure marginal. The plan's 70–80 GPU-h
came from scaling results/10's 65 s/structure at 398 aa with a 10 s/structure floor;
the real floor is lower.

| | planned | measured |
|---|---|---|
| total | 70–80 GPU-h | **~42 GPU-h** (256 × 9.9 min) |
| wall clock at `%3` | ~1 day | **~14 h** |

Disk on the cluster, projected from the first shard: MSAs 6.4 GB (the bulk), queries
0.2 GB, slim 0.23 MB/structure → **~5.8 GB**, total **~12.4 GB**. `/grupos` has 347 GB
free. `boltz_raw_output` is 16 KB — incremental per-shard slim is deleting raw
correctly (`delete_raw`). 0 failed shards so far.

The local slim store to sync back will be ~5.8 GB against 101 GB free here.

### 2026-08-26 — Phase 0 gate PASSED: their LOPO baselines reproduce

`rf4mave.py` on their own `preprocessed.pkl`, all 39 datasets / 29 proteins:

| model | features | ours | published | Δ |
|---|---|---|---|---|
| null (s̃_exp) | 3 | 0.334 | 0.17 | **+0.164** |
| ΔΔG only (Rosetta) | 1 | **0.249** | 0.25 | −0.001 |
| ΔΔE only (GEMME) | 1 | **0.409** | 0.42 | −0.011 |
| ΔΔG + ΔΔE | 2 | **0.466** | 0.47 | −0.004 |
| position-context | **47** | **0.519** | 0.52 | −0.001 |

**All four baselines pinned by explicit `-f` regexes in their `train.sh` reproduce
within ±0.011** — inside the ±0.02 gate. The harness is trustworthy. The
position-context set came out at exactly **47 features**, matching the paper's stated
count, which independently confirms the decoding (20+1+1 Rosetta, 20+1+1 GEMME, 3 s̃).

The null is the one outlier (+0.164). It is also the only one of the five that their
`train.sh` does **not** define with an explicit feature regex, so what went into their
Figure 2B green box is a guess on our side; the paper quotes it as a *mean*, not a
median. Our 0.334 agrees closely with their own Table S1 "MAVE WT→Mut" column
(median ≈ 0.33), i.e. with the substitution matrix used directly as a predictor. Read
as a definitional difference, not a harness bug — but it is a guess, and is reported
as one. It does not affect the ΔΔG comparison, which is what this experiment is for.

Also worth recording, since the paper leaves these vague and their code settles them:
RF is `n_estimators=150, max_features="sqrt", min_samples_leaf=15`; missing values are
a **−100 sentinel**, not NaN; their `-x 2` drops rows whose own Rosetta *or* GEMME value
is missing, from train and validation alike; and the 47 "position-context" features
decode exactly as 20 + 1 + 1 Rosetta, 20 + 1 + 1 GEMME, and 3 s̃ terms.

`check_frames.py` will verify the other half of the harness — that `score.py`'s rebuild
of those 47 features from the raw PRISM tables (needed so our ΔΔG can be swapped in)
reproduces their feature semantics, by running the same LOPO on both and comparing
per-dataset ρ.

### 2026-08-26 — GPU chain submitted (1414 → 1415 → 1416 → 1417)

`./slurm/submit_scan.sh experiment_configs/mave_hoie_le200.yaml 256 3` on cranex:
prepare **1414** (cpu) → predict array **1415** (`0-255%3`, gpu, self-slimming) →
slim sweep **1416** (`afterany`) → features **1417**. Bad nodes excluded up front:
cpu `nodo1,nodo3,nodo5`; gpu `nodo1,nodo3,nodo5,nodo11,nodo12,sauron`. Queue was
empty and GPU nodes idle at submission, so `%3` costs other users nothing.

256 shards ≈ 99 structures each. Expect prepare ~3–5 h (25k mutated MSAs; GLA's 7.5k
took 2 h), then ~70–80 GPU-h of predict.

**Deviation from the plan, stated deliberately:** the plan gated GPU submission on
Phase 0 finishing. Phase 0's random forests are slower than expected (~70 s per fold
for the single-feature models → ~4–6 h total), and Phase 0 validates the *scoring
harness*, which is not needed until Phase 3. The corpus itself was validated
independently through the real code path (`load_dataset` + `prepare_mutations_frame`:
25,213 rows in, **0 dropped**, 11 proteins, exactly 19 mutations per position,
25,224 structures). If Phase 0 turns out to need fixing, that changes `rf4mave.py`,
not the embeddings — so overlapping the two wastes nothing and saves a day.

**Cluster sync:** the cluster checkout is on `main` while this work is on branch
`results/11-12-calibration-and-error-anatomy`. Rather than switch its branch, the two
files the GPU run actually needs were written straight out of the pushed branch:
`git show origin/<branch>:<path> > <path>` for
`experiment_configs/mave_hoie_le200.yaml` and `data/raw/mave_hoie_le200.csv`. This
touches neither the cluster's branch nor its index. The `results/15/` scripts run
locally and are not needed there.

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
