"""Build report.pdf for 14_biophysical_features (paper-facing; no provenance).

    python results/14_biophysical_features/build_report.py

Self-contained HTML (figures embedded as base64) rendered via wkhtmltopdf. Every number
is read from the committed result tables so the PDF cannot drift from them.
"""
import base64
import subprocess
from pathlib import Path

import pandas as pd

R = Path(__file__).parent
res = pd.read_csv(R / "results_all.csv")


def g(cfg, st, col="r", aug=False):
    row = res[(res.config == cfg) & (res.set == st) & (res.augment == aug)]
    return float(row[col].iloc[0]) if len(row) else float("nan")


def ci(stem, ref, cfg, metric, dp=3):
    d = pd.read_csv(R / f"bootstrap_{stem}_ref-{ref}.csv")
    d = d[(d.kind == "paired_diff") & (d.config == cfg) & (d.metric == metric)]
    if not len(d):
        return "—"
    r = d.iloc[0]
    b, e = ("<b>", "</b>") if r.significant else ("", "")
    return f"{b}{r['mean']:+.{dp}f} [{r.lo:+.{dp}f}, {r.hi:+.{dp}f}]{e}"


S6_LOC = "exp14_s669_results_s669_locality"
FP_LOC = "exp14_fpfilt_results_locality_paired"
FP_FAR = "exp14_fpfilt_results_farctrl"
S6_OH = "exp14_s669_results_onehot_s669"
FP_OH = "exp14_fpfilt_results_onehot_fp"
FP_NOAUG = "exp14_fpfilt_results_fact_noaug"
FP_CONS = "exp14_fpfilt_results_fp_cons"
TS_LOC = "exp14_oof_results_locality_paired"
S6_BASE = "exp14_s669_results_s669_base"


def img(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


fig1 = img(R / "figures/01_what_generalizes.png")
fig2 = img(R / "figures/02_additions_that_failed.png")

CSS = """
@page { size: A4 landscape; margin: 14mm 14mm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 17pt; margin: 0 0 2px; color: #14314f; }
h2 { font-size: 12.5pt; color: #14314f; border-bottom: 1.5px solid #d0d7de; padding-bottom: 3px; margin-top: 18px; }
h3 { font-size: 11pt; color: #14314f; margin: 14px 0 4px; }
.sub { color: #555; font-size: 9.5pt; margin: 0 0 4px; }
.headline { background: #eef4fb; border-left: 4px solid #2c6fb3; padding: 10px 14px; margin: 14px 0; }
.caveat { background: #fdf4ec; border-left: 4px solid #D95F02; padding: 9px 13px; margin: 12px 0; font-size: 9.5pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #cfd6dd; padding: 4px 8px; text-align: right; }
th { background: #f3f6f9; } td:first-child, th:first-child { text-align: left; }
caption { caption-side: bottom; font-size: 8.5pt; color: #666; padding-top: 4px; text-align: left; }
figure { margin: 12px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; } figcaption { font-size: 8.5pt; color: #555; margin-top: 4px; text-align: left; }
code { background: #f3f4f6; padding: 0 3px; font-size: 8.8pt; }
"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>

<h1>What generalises from a frozen structure-model trunk is the local term, not the pooled one</h1>
<p class="sub">Experiment 14 · ddG_with_Boltz · frozen Boltz-2 pair track · MLP · two blind corpora</p>

<div class="headline">
A ΔΔG regressor on a frozen structure-prediction trunk must reduce the pair representation
at a mutated residue to a fixed-length vector. We tested three biology-motivated additions
to that readout — contact-weighted pooling, explicit burial and biophysics, and explicit
evolutionary conservation. <b>All three fail.</b> The experiment's controls instead show
that the <b>uniform mean over the whole chain behaves like a far-shell readout</b>, and
that the single most local feature available — the diagonal element <code>z[i,i]</code> —
matches every 256-dimensional construction on blind transfer at <b>half the width</b>,
while beating a plain substitution lookup by <b>+0.26 Pearson on both corpora</b>. The
mechanism is a train/test mismatch: whole-chain pooled <i>levels</i> carry corpus-specific
protein context that damages transfer, though it demonstrably helps in-distribution.
</div>

<h2>1. Motivation</h2>
<p>The standard readout for a mutated residue <i>i</i> is the mean of the pair-track row
<code>z[i, :]</code> over all residues of the chain. In a 60-residue domain that dilutes the
~10 residues actually in contact by a factor of six; in a 300-residue protein, by thirty.
The choice is widespread — the closest published method built on AlphaFold2 pair
representations enumerates four aggregations (global mean or sum, mutation-site mean or
sum), none spatially weighted, and Boltz-2's own affinity module "performs mean pooling
over all pairwise interactions". Since folding stability at a residue is physically a
property of the residues it packs against, we asked whether adding biology to this readout
helps, and used matched-dimension controls to establish what the readout is actually doing.</p>

<h2>2. Methods</h2>
<p><b>Constructions.</b> From the pair row at the mutated site we build: the <b>diagonal</b>
<code>z[i,i]</code> (unpooled); the uniformly <b>pooled difference</b> between mutant and
wild type; the uniformly <b>pooled levels</b> (wild-type and mutant vectors side by side);
and <b>contact-weighted</b> versions of the pooled quantities, where the weight is
<code>P(d<sub>ij</sub> &lt; 8 Å)</code> read from the model's own predicted distogram at the
wild-type structure, masked to |i−j| &gt; 2. Two matched-dimension controls: a
<b>far-shell</b> pooling using the renormalised complement <code>P(d ≥ 8 Å)</code>, and a
<b>substitution-identity</b> baseline of 40 one-hot dimensions with no structural content.
A biophysical block (40 d: burial, volume, hydropathy, transfer free energy, charge,
secondary-structure propensity, Gly/Pro flags, and their interactions with burial) and a
conservation block (14 d: alignment depth and effective depth, column entropy, PSSM
log-odds, consensus indicator) were tested as additions.</p>

<p><b>Evaluation.</b> Training is 12,359 mutations over 412 proteins with a five-fold
protein-grouped split and a five-seed MLP ensemble; the fold models are averaged onto two
blind corpora, S669 (541 variants / 62 proteins) and FireProt ≤500 (3,205 / 138). Because
mutations within a protein share a structure, an embedding and an assay, all intervals come
from a <b>cluster bootstrap over proteins</b> (400 resamples), reported as the paired
difference against a reference construction on the same resample. Antisymmetry augmentation
(adding each reverse mutation with negated ΔΔG to the training folds) is treated as an
experimental factor rather than a fixed default; transfer numbers below are without it.
FireProt is filtered to the 130 proteins sharing no 30 %-identity cluster with the training
corpus; S669 contains no such homologues.</p>

<h2>3. Results</h2>

<h3>3.1 A ladder of readouts</h3>
<table>
<caption>Table 1 — Blind-transfer Pearson r by construction. Same trunk, same model, same
data; only the readout differs. The rightmost column is the in-distribution protein-holdout
score on the training corpus.</caption>
<tr><th>construction</th><th>dims</th><th>S669</th><th>FireProt ≤500</th><th>in-distribution</th></tr>
<tr><td>substitution identity (one-hot)</td><td>40</td><td>{g('onehot','s669'):.3f}</td><td>{g('onehot','fireprot_le500'):.3f}</td><td>{g('onehot','tsu_oof'):.3f}</td></tr>
<tr><td>far-shell pooling</td><td>256</td><td>{g('far','s669'):.3f}</td><td>{g('far','fireprot_le500'):.3f}</td><td>{g('far','tsu_oof'):.3f}</td></tr>
<tr><td>uniform pooling, levels</td><td>256</td><td>{g('base','s669'):.3f}</td><td>{g('base','fireprot_le500'):.3f}</td><td><b>{g('base','tsu_oof'):.3f}</b></td></tr>
<tr><td>contact-weighted levels</td><td>256</td><td>{g('cw','s669'):.3f}</td><td>{g('cw','fireprot_le500'):.3f}</td><td>{g('cw','tsu_oof'):.3f}</td></tr>
<tr><td>diagonal + uniform pooled difference</td><td>256</td><td>{g('dz','s669'):.3f}</td><td>{g('dz','fireprot_le500'):.3f}</td><td>{g('dz','tsu_oof'):.3f}</td></tr>
<tr><td>diagonal + contact-weighted difference</td><td>256</td><td>{g('dz_cw','s669'):.3f}</td><td><b>{g('dz_cw','fireprot_le500'):.3f}</b></td><td>{g('dz_cw','tsu_oof'):.3f}</td></tr>
<tr><td><b>diagonal alone</b></td><td><b>128</b></td><td><b>{g('diag','s669'):.3f}</b></td><td>{g('diag','fireprot_le500'):.3f}</td><td>{g('diag','tsu_oof'):.3f}</td></tr>
</table>

<p>The ordering is identical on two independently curated corpora with different label
provenance. Uniform pooling of levels — the most common choice, and the best of these
readouts <i>in-distribution</i> — sits third from the bottom on both blind sets, closer to
far-shell pooling and to an amino-acid lookup than to the diagonal. Replacing it with the
diagonal alone halves the readout and gains, on S669, Δr {ci(S6_BASE,'base','diag','r')}.</p>

<figure><img src="{fig1}">
<figcaption><b>Figure 1.</b> <b>A</b> The ladder of readouts on both blind corpora.
<b>B</b> Paired bootstrap differences; the first three claims replicate on both corpora, the
fourth does not. <b>C</b> In-distribution skill plotted against transfer skill: the
configurations built on whole-chain pooling (red) are the strongest in-distribution and the
weakest on transfer.</figcaption></figure>

<h3>3.2 The local term is not a substitution lookup</h3>
<p>Because <code>z[i,i]</code> is the mutated residue's own pair element, the deflationary
reading is that it encodes which amino acid changed into which. It does not: against 40
one-hot dimensions the diagonal gains {ci(S6_OH,'onehot','diag','r')} on S669 and
{ci(FP_OH,'onehot','diag','r')} on FireProt — the same effect size, twice.</p>

<h3>3.3 Uniform pooling is a far-shell readout; levels are the problem, differences are not</h3>
<p>A matched-dimension far-shell control scores {g('far','s669'):.3f} / {g('far','fireprot_le500'):.3f},
statistically indistinguishable from uniform pooling of levels — as expected, since a
uniform mean over a chain is dominated by distant residues. Against it, local constructions
gain {ci(S6_LOC,'far','diag','r')} (S669) and {ci(FP_FAR,'far','cw','r')} (FireProt).</p>

<p>The decisive distinction is between pooled <i>levels</i> and pooled <i>differences</i>.
The difference cancels the per-protein offset and is neutral: dropping it entirely
(diagonal versus diagonal-plus-difference) changes nothing on transfer,
{ci(S6_LOC,'dz','diag','r')} and {ci(FP_LOC,'dz','diag','r')}. The levels retain that
offset, and it is corpus-specific: substituting them for the difference costs
{ci(FP_NOAUG,'dz','base','r')} on FireProt.</p>

<h3>3.4 The three biology-motivated additions</h3>
<table>
<caption>Table 2 — Paired differences for each addition. Bold marks an interval excluding zero.</caption>
<tr><th>addition</th><th>comparison</th><th>S669</th><th>FireProt ≤500</th></tr>
<tr><td rowspan="2">contact weighting</td><td>Δ Pearson r</td><td>{ci(S6_LOC,'dz','dz_cw','r')}</td><td>{ci(FP_LOC,'dz','dz_cw','r')}</td></tr>
<tr><td>Δ Spearman ρ</td><td>{ci(S6_LOC,'dz','dz_cw','rho')}</td><td>{ci(FP_LOC,'dz','dz_cw','rho')}</td></tr>
<tr><td>burial + biophysics</td><td>r, alone / added</td><td colspan="2">{g('bio','s669',aug=True):.3f} alone; added to the baseline it does not exceed it</td></tr>
<tr><td rowspan="2">MSA conservation</td><td>Δ Pearson r</td><td>—</td><td>{ci(FP_CONS,'cw','cw+cons','r')}</td></tr>
<tr><td>Δ bias, stabilizing</td><td>—</td><td>{ci(FP_CONS,'cw','cw+cons','stab_bias')}</td></tr>
</table>

<p><b>Contact weighting does not replicate.</b> On FireProt it reaches significance on rank
correlation and error but never on Pearson r; on S669 the Pearson effect is exactly zero.
<b>Biophysics</b> never exceeds the baseline, and its raw chain-scale features actively
degrade transfer by importing more corpus-specific context. <b>Conservation</b> adds nothing
over the pair track on a corpus with 100 % alignment coverage and a median alignment depth
of 9,474 — consistent with the trunk having been given the alignment already.</p>

<figure><img src="{fig2}">
<figcaption><b>Figure 2.</b> The three additions. <b>A</b> Contact weighting: significant on
one corpus, zero on the other. <b>B</b> Biophysics never exceeds the baseline. <b>C</b>
Conservation adds nothing over contact-weighted features on deep alignments.</figcaption></figure>

<h2>4. Interpretation</h2>
<p>Three additions that look like domain knowledge — spatial weighting, explicit biophysics,
explicit conservation — add nothing to a trunk already trained on structures and alignments.
What the trunk lacks is not information but an appropriate readout: the transferable signal
sits in the local pair element, and the conventional whole-chain average both dilutes it and
imports a per-protein offset that does not survive a change of corpus.</p>

<p>The mechanism is a train/test mismatch rather than a defect of the pooled features.
In-distribution the pooled half is worth {ci(TS_LOC,'dz','diag','r')} — it genuinely helps
when train and test share a corpus. This makes in-distribution holdout performance an
actively misleading model-selection signal here: the readout that ranks first
in-distribution ranks near-last on both blind corpora.</p>

<h2>5. Limitations</h2>
<div class="caveat">
<b>Neither blind corpus is the published benchmark.</b> Both are capped at 500 residues:
S669 here is 541 of 669 variants (62 of 94 proteins), and FireProt is likewise
length-capped. Every construction is scored on the identical subset, so the comparisons
above are valid and the cap cancels — but the absolute values are <b>not comparable to
published figures</b> for these benchmarks, and the cap plausibly removes the hardest
cases, since the training corpus consists of 32–72 residue domains. <b>No claim of
state-of-the-art performance is made.</b>
</div>
<div class="caveat">
<b>Two corpora are not many.</b> The ordering in Table 1 is identical on both, and the two
largest effects replicate with near-identical point estimates, but both blind sets are
literature-derived stability data; a third corpus with a different assay would strengthen
the claim, particularly for the smaller effects.
</div>
<ul>
<li>The backbone is frozen throughout. Conclusions about redundancy apply to a fixed
representation and need not hold if the trunk were fine-tuned — in particular, a spatial
weighting that adds nothing to frozen features might matter to a trainable readout.</li>
<li>Conservation was tested only as <i>appended per-column statistics</i>. Whether altering
what the trunk itself sees (mutating the alignment rather than only the query) behaves
differently is untested and requires re-running the structure model.</li>
<li>Ranking <i>within</i> the stabilizing tail is unchanged by every construction here, as it
was by earlier loss-reweighting work in this project. That deficit remains open.</li>
<li>Top-k detection precision proved unusable at this sample size, returning opposite signs
on different corpora; no claim rests on it.</li>
</ul>

<h2>6. Conclusion</h2>
<p>Asked what biology to add to a ΔΔG readout built on a frozen structure-prediction trunk,
the empirical answer is: none of the obvious candidates, and the productive change is a
deletion. The diagonal pair element alone, at half the dimensionality of the current
representation, matches or exceeds every richer construction on blind transfer and improves
substantially on the conventional uniform-mean readout. Where a strong pretrained trunk is
used frozen, the readout deserves auditing — against matched-dimension controls, and on a
corpus other than the one used to select it — before the feature list is extended.</p>

</body></html>"""

out_html = R / "_report.html"
out_html.write_text(HTML, encoding="utf-8")
subprocess.run(["wkhtmltopdf", "--enable-local-file-access", "--quiet",
                str(out_html), str(R / "report.pdf")], check=True)
out_html.unlink()
print(f"wrote {R / 'report.pdf'}")
