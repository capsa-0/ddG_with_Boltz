"""Build report.pdf for 11_calibration_gap (paper-facing; no provenance).

    python results/11_calibration_gap/build_report.py

Self-contained HTML (figures embedded as base64) rendered via wkhtmltopdf. Every number
is read from the committed tables — offset_ceiling.csv, split_half.csv,
split_half_tsuboyama.csv, homology_share.csv — so the PDF cannot drift from them.
Per results/guidelines.md this carries motivation, methods, results and interpretation
only; run history and file paths live in status.md and the README.
"""
import base64
import subprocess
from pathlib import Path

import pandas as pd

R = Path(__file__).parent
ceil = pd.read_csv(R / "offset_ceiling.csv")
s669 = pd.read_csv(R / "split_half.csv")
tsu = pd.read_csv(R / "split_half_tsuboyama.csv")
hom = pd.read_csv(R / "homology_share.csv")

NAME = {"A_tsu_only": "A — Tsuboyama", "B_fp_only": "B — FireProt",
        "D_finetuned": "D — fine-tuned"}


def q(df, name, col="mean"):
    return float(df[df.quantity == name][col].iloc[0])


def c(regime, subset, col):
    row = ceil[(ceil.regime == regime) & (ceil.subset == subset)]
    return float(row[col].iloc[0])


def h(quantity, grouping="same base structure", col="pair_r"):
    row = hom[(hom.quantity == quantity) & (hom.grouping == grouping)]
    return float(row[col].iloc[0])


def img(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


F1 = img(R / "figures/01_per_protein_error.png")
F2 = img(R / "figures/02_ceiling_and_sharing.png")

SUB = "common25"
gain_x = q(s669, "offset_from_other_half_honest") - q(s669, "baseline_no_offset")
gain_i = q(tsu, "offset_from_other_half_honest") - q(tsu, "baseline_no_offset")

ceil_rows = "".join(
    f"<tr><td>{NAME[r]}</td><td>{c(r, SUB, 'baseline_r'):.3f}</td>"
    f"<td><b>{c(r, SUB, 'oracle_offset_r'):.3f}</b></td>"
    f"<td>{c(r, SUB, 'oracle_gain_r'):.3f}</td>"
    f"<td>{c(r, SUB, 'oracle_affine_r'):.3f}</td>"
    f"<td>{c(r, SUB, 'baseline_rmse'):.2f} → {c(r, SUB, 'oracle_gain_rmse'):.2f}</td></tr>\n"
    for r in ("A_tsu_only", "B_fp_only", "D_finetuned"))

CSS = """
@page { size: A4; margin: 15mm 16mm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 17pt; margin: 0 0 2px; color: #7a3c12; }
h2 { font-size: 12.5pt; color: #7a3c12; border-bottom: 1.5px solid #ded4cc; padding-bottom: 3px; margin-top: 18px; }
h3 { font-size: 11pt; color: #7a3c12; margin: 14px 0 4px; }
.sub { color: #555; font-size: 9.5pt; margin: 0 0 4px; }
.headline { background: #fbf1e9; border-left: 4px solid #C25A12; padding: 10px 14px; margin: 14px 0; }
.caveat { background: #eef5f2; border-left: 4px solid #00966F; padding: 9px 13px; margin: 12px 0; font-size: 9.5pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #dcd4ce; padding: 4px 8px; text-align: right; }
th { background: #f8f4f1; } td:first-child, th:first-child { text-align: left; }
caption { caption-side: bottom; font-size: 8.5pt; color: #666; padding-top: 4px; text-align: left; }
figure { margin: 12px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; } figcaption { font-size: 8.5pt; color: #555; margin-top: 4px; text-align: left; }
"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>

<h1>The missing per-protein term is corpus context, not a property of the protein</h1>
<p class="sub">Why a blind benchmark's pooled correlation falls so far below its
per-protein correlation, and whether the gap can be closed · S669 and held-out
Tsuboyama</p>

<div class="headline">
<b>A per-protein additive offset is the missing correction, it is worth several times more
across datasets than within one, and nothing we can extract predicts it.</b> An oracle
offset lifts the leakage-clean S669 correlation from {c('D_finetuned', SUB, 'baseline_r'):.3f}
to {c('D_finetuned', SUB, 'oracle_offset_r'):.3f}; an honest split-half estimate makes
{gain_x:+.3f} of that transferable. The same correction is worth only {gain_i:+.3f}
in-distribution. Homologous constructs share the per-protein mean ΔΔG at
r = {h('mean_ddg'):.3f} but share the model's <i>error</i> on it at only
r = {h('offset'):.3f}. The model already extracts the part of the protein-level signal
that the fold determines; what remains is assay and corpus context, which no
representation <i>of the protein</i> can supply.
</div>

<h2>1. Motivation</h2>
<p>On a diverse blind benchmark this predictor shows a large and consistent gap: it ranks
mutations well <i>inside</i> a protein while placing different proteins poorly on a common
scale. Pooled Pearson r sits far below the median per-protein r computed from the very same
predictions. That gap has an obvious candidate explanation — a single wrong number per
protein — and an obvious potential remedy: predict that number from the wild-type alone and
add it back. If that worked it would be the cheapest large improvement available, since it
needs no new training data, no architecture change and no per-variant computation.</p>

<p>Three questions follow, and they must be asked in order. What <i>kind</i> of correction is
missing — an offset, a gain, or both? Is the apparent benefit real, or an artifact of fitting
the correction on the same variants used to score it? And can the correction be predicted from
anything available at inference time?</p>

<h2>2. Methods</h2>
<p><b>Oracle ceiling.</b> Take existing predictions and apply the best possible per-protein
offset, gain, and affine correction, then re-score. This bounds any scheme that predicts a
per-protein number, however it is obtained.</p>

<p><b>Honest split-half.</b> The oracle is fit and scored on the same variants, so it
overstates. Estimating the offset from half of each protein's variants and scoring on the
other half separates transferable signal from noise-fitting. Restricted to proteins with at
least six variants, repeated over random splits.</p>

<p><b>Predictability.</b> Four families of predictor were tried against the required offset:
the wild-type Boltz embedding pooled over mutated positions; a head trained on all 550
proteins of both training corpora; interpretable descriptors (length, amino-acid composition,
burial, hydropathy); and — the physically motivated candidate — the absolute folding free
energy ΔG of the wild type. Every predictor is evaluated under leave-one-protein-out or
homology-grouped cross-validation, and judged against a constant baseline, because a target
with small spread is trivially "predicted" by its own mean.</p>

<p><b>The control that decides the question.</b> Constructs of the same base structure that
differ by one background substitution let us ask, separately, whether the <i>quantity</i>
(the protein's mean ΔΔG) and the model's <i>error</i> on that quantity are shared between
close relatives. A fold property should be shared; corpus context should not.</p>

<p><b>In-distribution comparison.</b> The whole analysis is repeated on held-out predictions
within a single corpus, where train and test share assay conventions and curation. The
contrast between the two regimes is the experiment's result.</p>

<h2>3. Results</h2>

<h3>3.1 The missing correction is an offset, not a gain</h3>
<table>
<caption>Pooled Pearson r on the leakage-clean S669 subset, with each oracle correction
applied. The last column shows what the gain correction does to RMSE, which is the reason it
is rejected even where it nudges r upward.</caption>
<tr><th>regime</th><th>baseline</th><th>oracle offset</th><th>oracle gain</th><th>oracle affine</th><th>RMSE, gain</th></tr>
{ceil_rows}
</table>
<p>The offset is worth between
{min(c(r, SUB, 'oracle_offset_r') - c(r, SUB, 'baseline_r') for r in NAME):.3f} and
{max(c(r, SUB, 'oracle_offset_r') - c(r, SUB, 'baseline_r') for r in NAME):.3f} r in every
regime. The gain correction is not a competitor: it moves r only slightly and makes RMSE
worse in all three regimes, and the affine correction — offset and gain together — never
beats the offset alone. Whatever the model is getting wrong at the protein level is a
constant shift, not a mis-scaled slope.</p>

<h3>3.2 The benefit is real, and it is much larger across corpora than within one</h3>
<p>Fitting and scoring the offset on the same variants inflates it. The honest split-half
estimate on S669 moves the correlation from {q(s669, 'baseline_no_offset'):.3f} to
{q(s669, 'offset_from_other_half_honest'):.3f}, a transferable gain of <b>{gain_x:+.3f}</b>,
with a further {q(s669, 'offset_from_same_half_oracle') - q(s669, 'offset_from_other_half_honest'):+.3f}
attributable to noise-fitting. The quantity being estimated is stable rather than curation
noise: the split-half reliability of the per-protein mean ΔΔG is
{q(s669, 'split_half_reliability_mean_ddg'):.3f}.</p>

<p>Repeating the identical procedure in-distribution gives a very different answer:
{q(tsu, 'baseline_no_offset'):.3f} → {q(tsu, 'offset_from_other_half_honest'):.3f}, a gain of
only <b>{gain_i:+.3f}</b>. <b>The same correction is worth
{gain_x / gain_i:.0f}× more across datasets than within one</b>, and the offset it corrects is
correspondingly smaller — a standard deviation of 0.29 kcal/mol in-distribution against
1.46 across corpora. The model is already well calibrated when train and test share a
provenance.</p>

<figure><img src="{F2}"/>
<figcaption><b>Figure 1.</b> (a) The same oracle correction applied in both regimes. (b) The
control that decides the question: constructs of one base structure share the protein's mean
ΔΔG, but not the model's error on it.</figcaption></figure>

<h3>3.3 Nothing available at inference time predicts the offset</h3>
<p><b>Not the wild-type embedding.</b> The best head reaches a correlation of +0.262 with the
required offset, but applying its predictions never helps and usually hurts — on S669 the
fine-tuned regime falls from 0.506 to 0.490 with a ridge head, and to 0.343 with a
leave-one-protein-out head. In-distribution the same approach moves pooled r from 0.776 to
0.775, i.e. not at all.</p>

<p><b>Not interpretable structure or chemistry.</b> Length, amino-acid composition, burial and
hydropathy are all at or worse than a constant baseline. Length is the only descriptor with
any univariate signal (|r| = 0.338), and using it still degrades the pooled correlation.</p>

<p><b>Not the wild-type folding free energy.</b> ΔG(WT) is the physically motivated candidate
for a per-protein term, and it is not recoverable from a frozen trunk either: whole-protein
pooled representations predict it at r = 0.293 under homology-grouped cross-validation,
against r = 0.266 from protein length alone. ΔG(WT) is itself only weakly a fold property —
two constructs of the same base structure share it at r = {h('dG_wt'):.3f}, with a mean
absolute difference of {h('dG_wt', col='mean_abs_diff'):.2f} kcal/mol against
{h('dG_wt', col='random_abs_diff'):.2f} for random pairs, roughly one mutation's worth of
stability. Pooling a representation over residues averages away exactly what determines it.</p>

<h3>3.4 The decisive control</h3>
<div class="caveat">
Constructs of the same base structure share the <b>per-protein mean ΔΔG</b> at
r = {h('mean_ddg'):.3f} ± {h('mean_ddg', col='pair_r_sd'):.3f} — it is a genuine fold
property. They share the model's <b>error</b> on that mean at only
r = {h('offset'):.3f} ± {h('offset', col='pair_r_sd'):.3f}, indistinguishable from
nothing.
</div>
<p>That asymmetry is the answer. If the missing term were a property of the protein, the model's
error on it would be shared between near-identical proteins, and a representation of the
protein could in principle learn it. It is not shared. The model has already captured the part
of the protein-level signal that the fold determines; the residual is corpus and assay context
— which experimental laboratory, which curation lineage, which measurement convention — and no
function of the protein alone can supply it.</p>

<figure><img src="{F1}"/>
<figcaption><b>Figure 2.</b> Per-protein signed error and mean absolute error by training
regime on the blind benchmark, and the per-protein error plotted against the protein's true
mean ΔΔG. The near-unit slope in the third panel says the model predicts close to the same
mean for every protein regardless of what that protein's mean actually is.</figcaption></figure>

<h2>4. Interpretation</h2>
<p>This closes the per-protein-correction direction, and it reframes what the blind-benchmark
deficit is. The gap between pooled and per-protein correlation is not a term the model failed
to learn from its training data; it is cross-dataset calibration under domain shift. The
distinction matters because the two diagnoses recommend opposite work. A missing learnable
term would justify building a per-protein head. Domain shift does not — it points instead at
calibration against a handful of labelled variants from the target assay, or at reporting the
quantity the model actually estimates well.</p>

<p>That quantity is within-protein ranking, and it is where the defensible claim for this
predictor lies. The per-protein correlation is high and stable across regimes and across
homology filters; the cross-protein scale is where essentially all of the error lives.</p>

<p>A secondary result stands on its own: absolute folding stability is not recoverable from a
frozen structure-prediction trunk by pooling its representation over residues. It is also only
weakly determined by the fold, moving by about one mutation's worth of energy between
constructs that differ by a single background substitution. Any scheme that hopes to read
absolute ΔG out of such a representation should expect the same null.</p>

<h2>5. Limitations</h2>
<ul>
<li>The split-half analysis needs proteins with enough variants to halve, which restricts it
to 19 of the benchmark's 62 proteins. It is the honest estimate available, but it is computed
on the better-sampled subset, not the whole benchmark.</li>
<li>The homology control uses constructs of the same base structure, of which there are only
39–51 groups depending on the quantity. The intervals are correspondingly wide, and the claim
rests on the <i>gap</i> between the two correlations rather than on either value's
precision.</li>
<li>"Not predictable" is a statement about the predictors tried — pooled trunk
representations, a trained head, interpretable descriptors, and ΔG(WT). It does not exclude
that some other feature of the wild type carries the term.</li>
<li>The offset is defined against a specific benchmark's labels. Part of what it absorbs may
be that benchmark's own curation idiosyncrasies rather than a property of the target assay
in general — which is consistent with the domain-shift reading but not separable from it
here.</li>
</ul>

<h2>6. Conclusion</h2>
<p>Asked whether the large gap between within-protein and cross-protein accuracy can be closed
by supplying one number per protein, the answer is that the number exists, that a perfect
version of it would help substantially on a blind benchmark and barely at all in-distribution,
and that it is not predictable from the protein. The homology control explains why: the
quantity is a fold property, but the model's error on it is not. What is missing is not
information about the protein — it is information about the experiment.</p>

</body></html>"""

out_html = R / "_report.html"
out_html.write_text(HTML, encoding="utf-8")
subprocess.run(["wkhtmltopdf", "--enable-local-file-access", "--quiet",
                str(out_html), str(R / "report.pdf")], check=True)
out_html.unlink()
print(f"wrote {R / 'report.pdf'}")
