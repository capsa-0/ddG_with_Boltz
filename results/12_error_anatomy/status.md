# Status — 12_error_anatomy

**State:** 🚧 In progress
**Last updated:** 2026-08-27

## Current state

Which mutations does the raw-Δz predictor actually get wrong? Run on two held-out
sets — the S669 blind benchmark (541 variants, regime D) and **12,359 out-of-fold
Tsuboyama predictions** (5-fold GroupKFold on `wt_id`, the results/06 protocol). All
error tables are computed on the **protein-centred** error as well as the raw error, so
class effects are not swamped by the per-protein calibration gap studied in results/11.

Burial is derived from **Boltz's own predicted distogram** (`pdrow` in the slim store,
64 bins over 2–22 Å → expected residues within 10 Å, |i−j|>2), so no external
structures or DSSP are needed. Resolved for 91 % of Tsuboyama and 100 % of S669.

Headline: **most apparent class effects are effect-size artifacts.** After normalising
MAE by the class's own spread of true ΔΔG, burial is flat and residue identity nearly
flat. Only three real weak spots survive: **→Pro**, **buried glycines**, and — by far
the largest — **stabilizing mutations**.

All numbers are in the Log below; the analysis completed and printed every table.

## Next steps

- [x] Re-run `tsu_class_error.py` to regenerate `tsu_mut_classes.csv` and the Tsuboyama
      figure (the first run crashed on a matplotlib legend bug **after** printing all
      tables; script has since been patched to save the CSV before plotting).
- [x] Add `figures/README.md` index; add the Tsuboyama figure as `02_*`.
- [x] **Transfer pass** (2026-08-27): `transfer_class_error.py`, figure `03_*`, README
      section "The transfer pass" + the per-residue-ranking correction.
- [ ] `details.md` + `build_report.py` → `report.pdf`. Must carry the transfer pass, and
      must NOT reproduce the retracted per-residue error ranking.
- [x] **Follow-up experiment (the actionable one):** balanced-MSE loss reweighting for
      the stabilizing tail, following constraint-aware SPURS (arXiv 2606.08100). Done as
      results/13 — the tail is unmoved by loss reweighting alone.

## Blockers

None. Runs locally in ~15–25 min.

## Log — newest first

### 2026-08-27 — la anatomia de error en TRANSFERENCIA, rehecha con el readout correcto

**Por que se rehizo.** El desglose por clase en transferencia de este folder se corrio con
el readout **concat + regime D**, que results/14 despues midio como el *peor* en
transferencia (S669 r 0.476 vs 0.557 del `zdiag`). Y sobre S669 no tenia potencia
(`from-Pro` n=5, `->Pro` n=6). **FireProt nunca se habia desglosado por clase**, siendo el
unico corpus ciego con potencia (3.102 variantes / 130 proteinas filtradas vs 541 / 62).

**Como.** Sin GPU y sin re-entrenar: los dumps por variante que dejo results/14
(`data/processed/_analysis/exp14_{s669,fpfilt}_results_*.csv`), columna `diag` (el readout
de transferencia adoptado, `labels.TRANSFER_BLOCKS`) mas la columna `onehot` (control de
identidad de sustitucion, 40 dims). Error crudo y **centrado por proteina** (saca el offset
de calibracion de results/11). CIs por **bootstrap de clusters sobre proteinas**, 600
resamples. Scripts en el scratchpad de la sesion (`class_error_transfer.py`, `pass2.py`,
`pass3.py`); si el resultado se promueve a figura hay que moverlos a este folder.

**Gotcha de datos:** S669 tiene **17 pares (proteina, mutacion) repetidos con ddG medido
distinto**. Un merge por clave infla 541 -> 575 filas; hay que unir **por posicion**. La
primera pasada tenia ese bug (numeros ~2 % corridos), corregido.

**Referencia global (readout `diag`):** S669 MAE 0.977, MAE/sd 0.601, r 0.561 ·
FireProt-filt MAE 0.811, MAE/sd 0.508, r 0.645.

**Gly/Pro replican en transferencia** — FireProt, Delta(MAE/sd) de la clase vs el resto:

| clase | n | Delta(MAE/sd) | CI 95 % |
|---|---|---|---|
| ->Pro | 80 | **+0.099** | [+0.045, +0.772] * |
| desde Pro | 82 | **+0.098** | [+0.022, +0.232] * |
| desde Gly | 157 | **+0.085** | [+0.005, +0.251] * |
| ->Gly | 274 | -0.058 | [-0.107, +0.014] |
| desde aromatico (FWY) | 352 | +0.016 | [-0.039, +0.111] |
| X->Ala | 723 | +0.004 | [-0.044, +0.041] |
| casi-isosterica (dVol<30) | 1529 | **+0.070** | [+0.026, +0.113] * |

Mismo patron que el Tsuboyama in-distribution de este folder (->Pro 0.55, from-Gly 0.51,
from-Pro 0.50, ->Gly 0.46, otras 0.44). **Precision util: `->Gly` NO es una clase debil** —
es de las mejores. Lo debil es *salir* de Gly y Pro en cualquier direccion.
Lo casi-isosterico sobrevive el control del piso: tambien empeora en **ranking**
(rho 0.589 vs 0.709 FireProt; 0.438 vs 0.638 S669), no solo en MAE/sd. Caveat: las dos
metricas comparten el mismo confounder (spread chico => el ruido experimental pesa mas).

**CORRECCION a este folder — el ranking por residuo NO replica.** Spearman entre S669 y
FireProt del MAE/sd por residuo WT = **+0.05 (p=0.84)**, k=17. La lista del log del
2026-08-25 ("peores WT en S669: C 1.85, W 1.78, N 1.59; mejores K 0.68, V 0.78") es
**ruido de 541 variantes** y no debe citarse. Lo que si replica entre corpus es la rho
intra-clase (+0.62, p=0.025).

**Hallazgo nuevo — el skill sobre el lookup de aminoacido.** `skill = 1 - MAE_diag/MAE_onehot`:
global **+0.224 [+0.167, +0.255]** en FireProt, +0.143 en S669. Por residuo WT los extremos
coinciden en ambos corpus: **Q (-0.29 S669 / -0.03 FP) y W (+0.02 / +0.04)** — el embedding
no le gana a una matriz de sustitucion — contra V (+0.28/+0.27), L (+0.25/+0.28),
A (+0.11/+0.34). Por par: Y->F -0.03 (n=64), W->A -0.01 (n=14), W->F +0.02 (n=24),
K->R +0.03 (n=15) contra L->A +0.35/+0.37 (n=58/30), I->A +0.20/+0.40, V->A +0.22/+0.29,
G->A +0.25/+0.25. El ranking completo del skill replica solo a +0.40 (p=0.11) — **defendibles
los extremos, no el orden**.

Lectura: el embedding cobra su valor en **empaquetamiento hidrofobico** (alifatico->Ala) y
degenera a matriz de sustitucion cuando la mutacion **conserva la quimica** (Y->F, W->F,
K->R). Y ninguna clase tiene skill negativo con CI que excluya el global: el deficit de
Gly/Pro **no** es un colapso de la representacion (skill +0.21, normal) sino de magnitud.

**Promovido el mismo dia.** Los tres pases del scratchpad se consolidaron en
`transfer_class_error.py` (un solo script, sin re-entrenar nada, sin GPU), que escribe
`transfer_class_tables.csv`, `transfer_class_bootstrap.csv` y
`figures/03_transfer_class_error.png` (A: contrastes por clase con CI; B: skill sobre el
one-hot por residuo WT en los dos corpus; C: idem por par de sustitucion). README
actualizado con la seccion "The transfer pass", la correccion del ranking por residuo, el
gotcha de las 17 claves repetidas de S669 y las filas nuevas de provenance;
`figures/README.md` reindexado.


### 2026-08-25 — error anatomy on S669 + 12,359 out-of-fold Tsuboyama

**Held-out Tsuboyama (OOF), overall:** r=0.777, ρ=0.784, MAE 0.44, n=12,359
(fold r: 0.825 / 0.688 / 0.801 / 0.803 / 0.796) — consistent with results/06's 0.792.
**FireProt-only model on all of Tsuboyama** (never saw any of it): r=0.604, ρ=0.654.

**Bias is ≈0 in every class on Tsuboyama** (−0.16 to +0.06 kcal/mol across all WT
residues, all mutant residues, all burial tertiles). Contrast S669, where per-class
bias ran to ±1.86 — that is the per-protein calibration failure (results/11), not a
mutation-class effect.

**Burial (Tsuboyama):** buried MAE 0.60 / mid 0.42 / exposed 0.35 — 1.7× worse buried,
same ratio as S669 (1.46 / 1.01 / 0.85). But sd(true) is 1.23 / 0.87 / 0.73, so
**MAE÷sd = 0.49 / 0.48 / 0.48 — flat**, and ρ is *highest* at buried sites
(0.79 vs 0.69 exposed). Burial does not degrade the model; it scales the errors.
This contradicts the common "worse on buried residues" claim for this representation.

**Gly/Pro (Tsuboyama), MAE ÷ sd(true):** →Pro **0.55** (n=446) · from-Gly 0.51 (782) ·
from-Pro 0.50 (330) · →Gly 0.46 (638) · other **0.44** (10,163). →Pro is the genuinely
worst class; from-Gly mildly worse.

**Burial × Gly:** buried from-Gly MAE 0.76, sd 1.18 → **0.64** (n=81) vs buried non-Gly
0.59/1.23 → 0.48 and exposed non-Gly 0.33/0.67 → 0.49. **Buried glycines are the worst
cell** — independently corroborating results/10, where FoldX blew up at buried glycines
in GLA; here Boltz's own error peaks at the same sites.

**Effect direction (Tsuboyama) — the real deficit:**
| direction | n | MAE | bias | ρ | sd(true) |
|---|---|---|---|---|---|
| stabilizing | 535 | 0.64 | **+0.56** | **0.27** | 0.30 |
| destabilizing | 6,120 | 0.54 | −0.21 | 0.71 | 0.89 |
| neutral | 5,704 | 0.32 | +0.13 | 0.36 | 0.25 |

The model systematically calls stabilizing mutations destabilizing, and its MAE is
**2× the class's own spread** — it cannot resolve them. This is the one weak spot that
is not an effect-size artifact, and it is the one that matters for protein engineering.

**Volume change (Tsuboyama):** smaller 0.47 / larger 0.44 / similar 0.41 — weak.

**S669 (regime D), for comparison:** overall r=0.462, ρ=0.451, MAE 1.11.
Burial: buried 1.46 / mid 1.01 / exposed 0.85 (sd 1.95 / 1.48 / 1.23).
Direction: stabilizing MAE 1.86, bias **+1.86**, ρ 0.11 (n=69); destabilizing 1.25,
bias −1.08, ρ 0.43; neutral 0.47. **[RETRACTED 2026-08-27 — see the transfer-pass
entry above: this per-residue ranking does not replicate on FireProt, Spearman
+0.05.]** Worst WT residues C (1.85), W (1.78), N (1.59);
best K (0.68), V (0.78). Gly/Pro classes on S669 have too few variants to read
(from-Pro n=5, →Pro n=6).

**Prior art check.** Literature expects larger errors mutating away from Gly or to Pro,
and slightly worse on buried than exposed. We reproduce the Gly/Pro part and
**refute the burial part** once effect size is controlled for. results/10's GLA scan
had found glycines *better* there (per-mutation ρ +0.532 at G sites vs +0.445) — that
is consistent, since results/10 measured ranking (ρ), not magnitude, and ρ is indeed
not degraded at Gly sites here either.

**Gotcha:** the first run printed all tables then died in the figure legend
(`Line2D` has no `set_sizes`) *before* writing the CSV, losing ~20 min of MLP fits.
Script now writes `tsu_mut_classes.csv` immediately after prediction, before analysis.

**Code:** `tsu_class_error.py` (held-out Tsuboyama), `mut_class_error.py` (S669).
Intermediates land in `data/processed/_analysis/` (gitignored).
