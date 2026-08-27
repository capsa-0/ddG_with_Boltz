# Status — 11_calibration_gap

**State:** ✅ Done
**Last updated:** 2026-08-27

## Current state

Diagnostic experiment, run entirely on the **local workstation** (all inputs were
already in `data/processed/`; no cluster job was needed). It asks why the S669 pooled
Pearson (0.45 filtered) is so much lower than the per-protein median (0.54–0.59), and
whether the missing piece — a per-protein additive offset — can be predicted.

**Final answer (2026-08-25, numbers corrected 2026-08-27): the offset is a domain-shift
term, not a protein property.** It is worth **+0.144** pooled r on S669 but only **+0.029**
on held-out Tsuboyama, where the model is already well calibrated (offset sd 0.29 vs
1.46 kcal/mol on S669). And
while the per-protein **mean ΔΔG** *is* shared between homologues (pair r = 0.52 for
constructs of the same base structure), the model's **error** on it is **not**
(pair r = 0.09 ± 0.24) — i.e. the model already captures the part of the protein-level
signal that is a property of the fold. What remains is assay/corpus context, which no
representation of the protein can supply.

Nothing we can extract predicts the offset, on either dataset.

- A perfect per-protein offset is worth **+0.144 pooled r** (honest split-half; the
  in-sample oracle overstates it by +0.055). Per-protein *gain* correction does **not**
  help — it worsens RMSE in every regime, and offset alone beats affine.
- Split-half reliability of the per-protein mean ΔΔG on S669 (proteins with ≥6
  variants) is **0.823**, so the offset is a stable quantity, not curation noise.
- The offset is **not predictable** from: the WT Boltz embedding pooled over mutated
  positions (LOPO r=0.09/0.14; head trained on 550 proteins r=+0.262 best, applying it
  moves pooled r by +0.006 and *hurts* regime D); protein length, amino-acid
  composition, burial, or hydropathy (every set at or worse than a constant baseline;
  length is the only univariate signal at r=+0.32).
- **ΔG(WT)** — the natural physical candidate for the offset — is also **not**
  recoverable from the whole-protein Boltz single representation: r=+0.29 / ρ=+0.30
  under 30 %-identity GroupKFold, against **r=+0.27 from protein length alone**.
- ΔG(WT) is **not a fold-level property**: two constructs of the *same base structure*
  differing by one background substitution share it at only r≈0.32, mean |Δ| 0.64
  kcal/mol (random pairs 1.07). It moves by roughly one mutation's worth of stability
  per substitution — so pooling a representation over residues averages away exactly
  what determines it.

Interpretation for the write-up: **the Boltz trunk representation ranks mutations well
within a fold and carries no protein-level stability scale.** That single statement
explains the 0.61-vs-0.40 gap, the regime-A calibration failure in results/09, and the
ΔG null here.

## Next steps

- [x] Scope settled: this folder = calibration gap + offset predictability + ΔG(WT).
      Error anatomy split out to `results/12_error_anatomy`; the SOTA scan lives in
      `theory/sota_2026.md`.
- [x] `exp1_offset.py` on held-out Tsuboyama — **done**, and it is what settled the
      question (offset ceiling only +0.029 there).
- [x] Homology-sharing table for the offset — **done** (offset is *not* shared; the
      mean ΔΔG is).
- [x] Scripts ported into this folder; they now write to `data/processed/_analysis/`.
- [ ] Add `figures/README.md`; add a figure for the S669-vs-Tsuboyama offset contrast
      (the one panel that carries the final conclusion).
- [ ] `details.md` + `build_report.py` → `report.pdf`.
- [ ] **Do not** pursue a per-protein correction head — this folder closes that
      direction. The remaining protein-level idea (per-residue ΔG, `ΔG = Σ_i g(s_i)`,
      as IFUM does) is a *different* architecture and belongs in its own experiment;
      note that ΔG(WT) itself is only weakly recoverable (r 0.29), so that path starts
      from a weak prior.

## Blockers

None. All inputs are local; runtimes are ~10–25 min per script on the workstation.

## Log — newest first

### 2026-08-27 — report.pdf + figura 02; **tres bugs y el estimador defectuoso**

El folder no tenía **ninguna tabla comprometida**: sus números vivían solo en la prosa del
README. Al intentar generarlas aparecieron tres problemas, todos arreglados.

**1. `offset_ceiling.py` nunca pudo guardar su tabla.** `NameError: name 'SCR' is not
defined` en la última línea — todos los scripts hermanos definen `SCR`, este no. Calculaba
todo, lo imprimía y moría al escribir, así que el CSV **nunca existió**. Ahora escribe
`offset_ceiling.csv` en el folder (no en el scratch gitignorado).

**2. El defecto del estimador estaba acá también**, y llegaba por dos caminos:
- directo: `offset_ceiling.py` y `offset_learn.py` con `max_iter=250, early_stopping=False`;
- por cache: `offset_real.py` lee `s669_predictions.csv`, que escribe
  `results/12/mut_class_error.py` — también defectuoso, y el cache era del 08-25, anterior a
  la corrección del 08-27. **El defecto se propagaba entre carpetas.**
Corregidos los cuatro (más `results/12/tsu_class_error.py`, la quinta ocurrencia). Barrido
del repo: queda una viva en `results/08/run_finetune.py`, fuera de alcance acá.

**3. `offset_learn.py` escribía la figura al scratch gitignorado** y alguien la copiaba a
mano a `figures/`, así que una re-corrida dejaba la figura commiteada obsoleta en silencio.
Ahora escribe directo a `figures/01_per_protein_error.png`.

**Los números se mueven; la conclusión no.**

| | README decía | corregido |
|---|---|---|
| baseline régimen D (common25) | 0.408 | **0.453** |
| techo con offset oráculo | 0.651 | 0.643 |
| split-half honesto | 0.444 → 0.648 | **0.511 → 0.655** |
| **ganancia transferible** | **+0.204** | **+0.144** |
| mejor cabeza de offset | r=+0.185 | r=+0.262 (ridge, régimen B) |
| figura 01 panel 3 | r = 0.91 | r = 0.88 |

Lo que **no** se movió, porque no depende del estimador de S669: el contraste
in-distribution (+0.029, 0.779 → 0.808), ΔG(WT) 0.293 contra 0.266 de longitud, y **la fila
decisiva** — homólogas comparten la media ΔΔG (+0.516 ± 0.080) pero no el error del modelo
sobre ella (+0.090 ± 0.242).

Una afirmación hay que reescribirla: el README decía que el *gain* oráculo "perjudica a los
regímenes A y B". Con los números corregidos les sube levemente el r — pero **empeora el RMSE
en los tres regímenes** (D: 1.618 → 1.722), que es la razón real para descartarlo. Reescrito
así.

La razón cross/in-distribution pasó de ~7× a **~5×**. La conclusión (el offset es corrimiento
de dominio, no propiedad de la proteína) queda intacta.

**Salidas nuevas:** `offset_ceiling.csv`, `split_half.csv`, `split_half_tsuboyama.csv`,
`homology_share.csv` (los tres últimos por parches que hacen persistir lo que antes solo se
imprimía), `make_figures.py` → `figures/02_ceiling_and_sharing.png` — el pendiente
"figura contrastando los techos" que el folder arrastraba —, `figures/README.md`,
`build_report.py` → `report.pdf` (3 páginas, 2 figuras, 0 términos de procedencia).

**Nota de reproducibilidad:** correr con `conda run -n ddG_with_Boltz`. El python base del
workstation tiene sklearn 1.7.2, donde `MLPRegressor((256,64), ...)` posicional falla porque
el primer parámetro posicional pasó a ser `loss`; el env del proyecto tiene 1.6.1.

### 2026-08-25 (later) — the offset is domain shift, not a protein property

Two runs completed that reverse the provisional reading above.

**1. The offset ceiling collapses on held-out Tsuboyama** (`exp1_offset.py`;
11,189 muts / 373 proteins, median **30 mutations per protein**; OOF r=0.776, ρ=0.789):

| | S669 (median 3 muts/protein) | Tsuboyama (median 30) |
|---|---|---|
| baseline | 0.444 ± 0.035 | 0.779 ± 0.024 |
| + offset from other half (honest) | **0.648 ± 0.028** | **0.808 ± 0.021** |
| + offset from same half (in-sample) | 0.702 ± 0.027 | 0.831 ± 0.022 |
| **real transferable gain** | **+0.204** | **+0.029** |
| offset sd | 1.43 kcal/mol | **0.29 kcal/mol** |
| split-half reliability of mean ΔΔG | 0.823 | 0.637 |

In-distribution the model is **already calibrated** — the offset it needs is 5× smaller,
and correcting it perfectly buys almost nothing. The +0.204 on S669 is therefore a
property of the *cross-dataset transfer*, not a universal missing term.

**2. Predicting the offset on Tsuboyama is also null** (5-fold GroupKFold, 145 clusters
at 30 % identity, target sd 0.29): length r=−0.111 · length+composition −0.030 ·
**Boltz whole-protein `s` (ridge) +0.038** · same MLP +0.039 (RMSE 0.40 vs const 0.29) ·
`s` mean-pool only −0.010 · s+len+comp +0.019. Every variant leaves pooled r at
0.774–0.775 vs a 0.776 baseline; the MLP drops it to 0.747.

**3. Homology sharing — the decisive decomposition** (`homology_share.py`):

| quantity | same base structure | cluster 90 % | cluster 50 % | cluster 30 % |
|---|---|---|---|---|
| per-protein **mean ΔΔG** | **+0.516 ± 0.080** | +0.302 | +0.407 | +0.224 |
| per-protein **offset** (model error) | +0.090 ± 0.242 | +0.133 | +0.144 | +0.102 |
| **ΔG(WT)** | +0.315 ± 0.121 | +0.268 | +0.106 | +0.211 |

The protein-level ΔΔG mean **is** a fold property (r = 0.52 between constructs of the
same base structure). The model's **error** on it is **not** (r = 0.09, CI spanning
zero). Read together: *the model already extracts the shareable, fold-determined part of
the protein-level signal.* The residue is corpus/assay context — which is why no
protein representation predicts it, and why it appears only on cross-dataset transfer.

Reference correlations (n=373): ΔG(WT) vs per-protein mean ΔΔG **r=+0.464**;
ΔG(WT) vs offset r=+0.245. So ΔG(WT) does relate to mean ΔΔG as the physics predicts,
but explains only ~22 % of its variance — and is itself only weakly predictable (0.29).

**Consequence for the project:** the per-protein correction direction is **closed**. The
S669 pooled-r deficit should be reported as **cross-dataset calibration under domain
shift**, not as a missing protein-level term the model could learn. Within-protein
ranking (per-protein median r 0.58–0.61 on S669, ρ 0.789 in-distribution) is the honest
headline claim.

*Caveat:* the offset pair-r rows have wide error bars (±0.20–0.27, 42–63 groups); they
support "not detectably shared", not "provably zero".

### 2026-08-25 — diagnosis of the pooled-vs-per-protein gap; four independent nulls
- Motivated by a state-of-the-art scan (see session note below): our S669 filtered
  pooled r (0.404 B / 0.408 D) sits ~0.14 behind the leaders (Mutate Everything 0.56,
  constraint-aware SPURS 0.54), while our per-protein median (0.58–0.61) is competitive.
- **Variance decomposition, S669:** ΔΔG variance is 38.9 % between-protein /
  61.1 % within-protein; per-protein mean ΔΔG has sd 1.69 kcal/mol.
- **Oracle ceiling (common25 subset, n=360):** baseline → +offset / +gain / +affine
  = A 0.214→0.572/0.266/0.514 · B 0.404→0.599/0.321/0.496 · D 0.408→0.651/0.455/0.646.
  Offset alone dominates; gain alone *hurts* A and B.
- **Honest split-half oracle** (19 proteins with ≥6 variants, 442 variants, 200 reps):
  baseline 0.444±0.035 → offset-from-other-half **0.648±0.028** → offset-from-same-half
  0.702±0.027. Real transferable gain **+0.204**; noise-fitting inflation +0.055.
- **Offset predictability (all null).** LOPO ridge from protein-mean `wtz`:
  r=0.087 (B) / 0.135 (D), both worse than a constant; applying it drops pooled r
  0.500→0.411 and 0.462→0.305. Head trained on all 550 Tsuboyama+FireProt proteins:
  ridge r=+0.185 (B, pooled +0.006), RF r=+0.053; regime D goes negative
  (ridge −0.121, RF −0.222 → pooled 0.408→0.352).
  Interpretable descriptors (length / 20-aa composition / burial / chemistry): every
  set at or worse than the constant baseline (RMSE 1.39–1.46 vs const 1.43).
  *Note:* the LOO correlations printed as ≈ −1 for the no-signal sets are an artifact
  of ridge shrinking to the intercept, not anti-prediction — read the RMSE column.
- **Per-protein error figure** (S669, 3 regimes): per-protein mean signed error
  sd 1.41–1.55 kcal/mol, range −6.1 to +3.6; regime A has a global bias (mean −0.68).
  Per-protein MAE medians B 0.85 / D 1.05 / A 1.12. Panel 3: per-protein mean error vs
  the protein's true mean ΔΔG has **r=0.91, slope 0.77** — the model predicts nearly the
  same mean for every protein, so the required offset *is* ~0.77 × the protein's own
  mean ΔΔG. Worst cells are 1–4-variant proteins (O73951 n=2, P84131 n=1, P11053 n=1),
  which the oracle deliberately skipped (guard: n<3 not corrected).
- **New data wired up:** absolute ΔG(WT) from the Tsuboyama 2023 supplementary,
  found locally at
  `ddg_datasets/dms/Processed_K50_dG_datasets/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv`
  (column `dG_ML`, rows with `mut_type=='wt'`). **412/412** of our proteins match on
  `WT_name`. Extracted to scratch as `tsu_wt_dG.csv`. dG(WT): mean 2.74, sd 0.92,
  range 0.71–4.73 kcal/mol.
- **ΔG(WT) prediction (null):** 373 proteins with a WT slim entry, 5-fold GroupKFold on
  145 clusters at 30 % identity. length 0.266 · composition 0.105 · length+comp 0.279 ·
  **Boltz whole-protein `s` (ridge) 0.293** · same with MLP 0.109 · s+len+comp 0.299.
  RMSE 0.88–0.92 against a constant baseline of 0.92.
- **Homology sharing of ΔG(WT):** same base structure (39 groups) pair r=+0.315±0.121,
  |Δ|=0.64 (random 1.07); cluster 90 % r=+0.268; 50 % r=+0.106; 30 % r=+0.211.
  *Caveat:* the ICC printed alongside is inflated by singleton clusters — use pair r
  and |Δ|. n=39 groups → CI on r=0.32 is roughly [0.0, 0.58]; weak evidence.
- **Gotcha:** `data/raw/tsuboyama_bench_clusters.csv` is **degenerate** — all 412
  proteins are assigned to a single cluster (`7JJK.pdb`). Use
  `data/processed/tsuboyama_bench_fast/cluster_map_{30,50,90}.csv` instead
  (147 / 228 / 306 clusters). Worth fixing or deleting that raw file.
- **Scratch scripts** (to be ported into this folder):
  `offset_ceiling.py`, `offset_learn.py`, `offset_real.py`, `exp4_dG.py`,
  `exp1_offset.py`, `homology_share.py`, in this session's scratchpad.

### 2026-08-25 — session note: results from this session still need splitting
Three separate threads were produced in one session and are **not** all this folder's:
1. **this folder** — the calibration gap and the protein-level offset (above);
2. **error anatomy by mutation class** (S669 + 12,359 out-of-fold Tsuboyama; burial,
   Gly/Pro, effect direction) — proposed as a sibling result folder;
3. **a state-of-the-art literature scan** (AFToolkit, Mutate Everything, SPURS,
   constraint-aware SPURS, IFUM, SaProtΔG, Boltz-2-PPI, Boltz-1 trunk probing) —
   not a result; proposed for `theory/`.
Final split awaiting the user's decision.
