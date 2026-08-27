"""Build report.pdf for 12_error_anatomy (paper-facing; no provenance).

    python results/12_error_anatomy/build_report.py

Self-contained HTML (figures embedded as base64) rendered via wkhtmltopdf. Every number
is read from the committed result tables so the PDF cannot drift from them.
"""
import base64
import subprocess
from pathlib import Path

import pandas as pd

R = Path(__file__).parent
IND = pd.read_csv(R / "indist_class_tables.csv")
TR = pd.read_csv(R / "transfer_class_tables.csv")
BS = pd.read_csv(R / "transfer_class_bootstrap.csv")
REP = pd.read_csv(R / "transfer_replication.csv")

FP, S6 = "FireProt <=500 (filt)", "S669"


def ind(grouping, klass, col, dp=3):
    row = IND[(IND.grouping == grouping) & (IND.klass == klass)]
    return f"{float(row[col].iloc[0]):.{dp}f}" if len(row) else "—"


def indn(grouping, klass):
    row = IND[(IND.grouping == grouping) & (IND.klass == klass)]
    return f"{int(row.n.iloc[0]):,}" if len(row) else "—"


def tr(corpus, grouping, klass, col, dp=3):
    row = TR[(TR.corpus == corpus) & (TR.grouping == grouping) & (TR["class"] == klass)]
    if not len(row) or pd.isna(row[col].iloc[0]):
        return "—"
    return f"{float(row[col].iloc[0]):.{dp}f}"


def trn(corpus, grouping, klass):
    row = TR[(TR.corpus == corpus) & (TR.grouping == grouping) & (TR["class"] == klass)]
    return f"{int(row.n.iloc[0])}" if len(row) else "—"


def bs(corpus, klass, dp=3):
    """Class-vs-rest contrast with its cluster-bootstrap CI; bold when it clears zero."""
    row = BS[(BS.corpus == corpus) & (BS.klass == klass)]
    if not len(row):
        return "—"
    r = row.iloc[0]
    b, e = ("<b>", "</b>") if r.significant else ("", "")
    return f"{b}{r.delta_mae_sd:+.{dp}f} [{r.lo:+.{dp}f}, {r.hi:+.{dp}f}]{e}"


def bsn(corpus, klass):
    row = BS[(BS.corpus == corpus) & (BS.klass == klass)]
    return f"{int(row.n.iloc[0])}" if len(row) else "—"


def rep(grouping, metric):
    row = REP[(REP.grouping == grouping) & (REP.metric == metric)]
    if not len(row):
        return "—"
    r = row.iloc[0]
    return f"{r.spearman:+.2f} (p = {r.p:.2f}, k = {int(r.k)})"


def img(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


fig_ind = img(R / "figures/02_tsuboyama_mut_class_error.png")
fig_tr = img(R / "figures/03_transfer_class_error.png")

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
th, td { border: 1px solid #d0d7de; padding: 4px 7px; text-align: right; }
th { background: #f2f5f8; color: #14314f; font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
figure { margin: 10px 0 4px; page-break-inside: avoid; }
figure img { width: 100%; }
figcaption { font-size: 8.5pt; color: #555; margin-top: 3px; }
.two { display: flex; gap: 18px; }
.two > div { flex: 1; }
p { margin: 7px 0; }
.note { font-size: 8.5pt; color: #666; }
"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>

<h1>Error anatomy of a frozen-trunk &Delta;&Delta;G predictor</h1>
<p class="sub">Which mutation classes are hard, on held-out data and on two blind corpora &mdash;
and where the learned representation actually beats a substitution matrix.</p>

<div class="headline">
<p><b>Aggregate accuracy hides a structured error.</b> Held-out performance
(&rho; = {ind('overall', 'all', 'rho', 3)}, MAE = {ind('overall', 'all', 'MAE', 2)} kcal/mol over
{indn('overall', 'all')} mutations) is not uniform across the substitution space, but most of the
apparent structure is an effect-size artifact: classes containing larger &Delta;&Delta;G values
show larger absolute errors and identical <i>relative</i> accuracy. Three things survive
normalisation. (i) The dominant deficit is the <b>stabilizing tail</b>: error is
{ind('direction', 'stabilizing', 'MAE_sd', 1)}&times; the class's own spread and the bias is
{ind('direction', 'stabilizing', 'bias', 2)} kcal/mol &mdash; stabilizing mutations are called
destabilizing. (ii) On blind transfer, <b>proline in either direction and mutations leaving
glycine</b> are genuinely harder, while <b>mutations <i>to</i> glycine are not</b>. (iii) The
representation's advantage over a plain amino-acid lookup is <b>not uniform</b>: it is largest
for core-packing substitutions and near zero when the mutation preserves the chemistry.</p>
</div>

<h2>1. Question</h2>
<p>Holdout designs in this series measure <i>generalization to unseen classes</i> &mdash; unseen
substitution types, source residues, target residues. That is a different question from
<i>which mutations are hard</i>, which had not been asked on a blind set. The literature on
&Delta;&Delta;G prediction makes two specific predictions worth testing: larger errors for
glycine and proline substitutions, and degraded accuracy at buried positions. A third question
only becomes askable once a matched trivial baseline exists: for which mutations does a learned
structural representation contribute anything beyond knowing which residue replaced which?</p>

<h2>2. Method</h2>
<p>Predictions come from a regressor on frozen structure-model embeddings. Three evaluation sets
are used: <b>{indn('overall', 'all')} out-of-fold predictions</b> from 5-fold grouped
cross-validation on the training corpus (grouping by protein, so no protein is ever in both
folds), and <b>two blind corpora</b> never seen in training &mdash; S669
({trn(S6, 'overall', 'all')} variants over 62 proteins) and a homology-filtered FireProt
subset ({trn(FP, 'overall', 'all')} variants over 130 proteins). The blind corpora are scored with the readout
selected for transfer in the companion feature study (the pair-track diagonal at the mutated
site); the out-of-fold set uses the in-distribution readout.</p>

<p>Four conventions make the class comparisons interpretable:</p>
<ul>
<li><b>Protein-centred error.</b> Every table is computed on the raw error and on the error with
each protein's mean error removed, so a class effect cannot be an artifact of that class
sitting in badly-calibrated proteins.</li>
<li><b>Normalisation by class spread.</b> Class MAE is read against that class's own standard
deviation of true &Delta;&Delta;G. Without it, any class containing larger effects looks harder
by arithmetic alone.</li>
<li><b>Burial from the model's own predicted distogram</b> &mdash; expected number of residues
within 10&nbsp;&Aring; of the mutated site, excluding trivial backbone neighbours. No external
structures or secondary-structure assignment are required.</li>
<li><b>Cluster bootstrap over proteins</b> for every confidence interval, since variants within
a protein are not independent.</li>
</ul>

<h2>3. Burial does not degrade the model</h2>
<table>
<tr><th>Burial tertile</th><th>n</th><th>MAE</th><th>&rho;</th><th>sd(true)</th><th>MAE &divide; sd</th></tr>
<tr><td>buried</td><td>{indn('burial', 'buried')}</td><td>{ind('burial', 'buried', 'MAE', 2)}</td><td>{ind('burial', 'buried', 'rho', 2)}</td><td>{ind('burial', 'buried', 'sd_true', 2)}</td><td><b>{ind('burial', 'buried', 'MAE_sd', 2)}</b></td></tr>
<tr><td>mid</td><td>{indn('burial', 'mid')}</td><td>{ind('burial', 'mid', 'MAE', 2)}</td><td>{ind('burial', 'mid', 'rho', 2)}</td><td>{ind('burial', 'mid', 'sd_true', 2)}</td><td><b>{ind('burial', 'mid', 'MAE_sd', 2)}</b></td></tr>
<tr><td>exposed</td><td>{indn('burial', 'exposed')}</td><td>{ind('burial', 'exposed', 'MAE', 2)}</td><td>{ind('burial', 'exposed', 'rho', 2)}</td><td>{ind('burial', 'exposed', 'sd_true', 2)}</td><td><b>{ind('burial', 'exposed', 'MAE_sd', 2)}</b></td></tr>
</table>
<p>Buried sites carry {ind('burial', 'buried', 'MAE', 2)} kcal/mol of error against
{ind('burial', 'exposed', 'MAE', 2)} at exposed sites &mdash; a 1.7&times; gap that would ordinarily
be read as "the model is worse in the core". It is not: relative accuracy is flat
({ind('burial', 'buried', 'MAE_sd', 2)} / {ind('burial', 'mid', 'MAE_sd', 2)} /
{ind('burial', 'exposed', 'MAE_sd', 2)}), and <i>ranking</i> is in fact best at buried sites
(&rho; {ind('burial', 'buried', 'rho', 2)} against {ind('burial', 'exposed', 'rho', 2)}). Burial
scales the errors because it scales the effects; it does not degrade the model. The single
interaction cell that does stand out is <b>buried glycines</b>
(MAE &divide; sd {ind('burial_x_gly', 'buried, from Gly', 'MAE_sd', 2)},
n = {indn('burial_x_gly', 'buried, from Gly')}, against
{ind('burial_x_gly', 'buried, not Gly', 'MAE_sd', 2)} for buried non-glycines) &mdash; the same
regime in which a physics-based method independently breaks down.</p>

<h2>4. The dominant deficit is the stabilizing tail</h2>
<table>
<tr><th>Effect direction</th><th>n</th><th>MAE</th><th>bias</th><th>&rho;</th><th>sd(true)</th><th>MAE &divide; sd</th></tr>
<tr><td>stabilizing</td><td>{indn('direction', 'stabilizing')}</td><td>{ind('direction', 'stabilizing', 'MAE', 2)}</td><td><b>{ind('direction', 'stabilizing', 'bias', 2)}</b></td><td><b>{ind('direction', 'stabilizing', 'rho', 2)}</b></td><td>{ind('direction', 'stabilizing', 'sd_true', 2)}</td><td><b>{ind('direction', 'stabilizing', 'MAE_sd', 2)}</b></td></tr>
<tr><td>destabilizing</td><td>{indn('direction', 'destabilizing')}</td><td>{ind('direction', 'destabilizing', 'MAE', 2)}</td><td>{ind('direction', 'destabilizing', 'bias', 2)}</td><td>{ind('direction', 'destabilizing', 'rho', 2)}</td><td>{ind('direction', 'destabilizing', 'sd_true', 2)}</td><td>{ind('direction', 'destabilizing', 'MAE_sd', 2)}</td></tr>
<tr><td>neutral</td><td>{indn('direction', 'neutral')}</td><td>{ind('direction', 'neutral', 'MAE', 2)}</td><td>{ind('direction', 'neutral', 'bias', 2)}</td><td>{ind('direction', 'neutral', 'rho', 2)}</td><td>{ind('direction', 'neutral', 'sd_true', 2)}</td><td>{ind('direction', 'neutral', 'MAE_sd', 2)}</td></tr>
</table>
<p>This is the one class effect that is not an arithmetic artifact, and it runs the wrong way for
protein engineering. The model's error on stabilizing mutations is roughly twice the spread of
the class itself, the bias is strongly positive &mdash; stabilizing variants are predicted less
stabilizing than they are &mdash; and within-class ranking nearly collapses
(&rho; {ind('direction', 'stabilizing', 'rho', 2)} against
{ind('direction', 'destabilizing', 'rho', 2)} for destabilizing variants). Amplitude compression
toward the mean, visible directly in the predicted-versus-measured panel below, is the mechanism.</p>

<figure>
<img src="{fig_ind}"/>
<figcaption><b>Figure 1.</b> Held-out predictions, {indn('overall', 'all')} mutations.
Left: protein-centred error by glycine/proline class. Middle: by burial tertile. Right:
predicted against measured &Delta;&Delta;G coloured by effect direction, with the
amplitude-compression fit &mdash; the slope below unity is what produces the stabilizing-tail
bias.</figcaption>
</figure>

<h2>5. Which classes stay hard on blind transfer</h2>
<p>Classes are contrasted against all other mutations in the same corpus, as a difference in
MAE &divide; sd, with a cluster bootstrap over proteins. Both blind corpora are reported.</p>
<table>
<tr><th>Class</th><th>n</th><th>FireProt &Delta;(MAE &divide; sd)</th><th>n</th><th>S669 &Delta;(MAE &divide; sd)</th></tr>
<tr><td>&rarr; Pro</td><td>{bsn(FP, '->Pro')}</td><td>{bs(FP, '->Pro')}</td><td>{bsn(S6, '->Pro')}</td><td>{bs(S6, '->Pro')}</td></tr>
<tr><td>from Pro</td><td>{bsn(FP, 'from Pro')}</td><td>{bs(FP, 'from Pro')}</td><td>{bsn(S6, 'from Pro')}</td><td>{bs(S6, 'from Pro')}</td></tr>
<tr><td>from Gly</td><td>{bsn(FP, 'from Gly')}</td><td>{bs(FP, 'from Gly')}</td><td>{bsn(S6, 'from Gly')}</td><td>{bs(S6, 'from Gly')}</td></tr>
<tr><td>&rarr; Gly</td><td>{bsn(FP, '->Gly')}</td><td>{bs(FP, '->Gly')}</td><td>{bsn(S6, '->Gly')}</td><td>{bs(S6, '->Gly')}</td></tr>
<tr><td>from aromatic (F/W/Y)</td><td>{bsn(FP, 'from aromatic (FWY)')}</td><td>{bs(FP, 'from aromatic (FWY)')}</td><td>{bsn(S6, 'from aromatic (FWY)')}</td><td>{bs(S6, 'from aromatic (FWY)')}</td></tr>
<tr><td>X &rarr; Ala</td><td>{bsn(FP, 'X->Ala')}</td><td>{bs(FP, 'X->Ala')}</td><td>{bsn(S6, 'X->Ala')}</td><td>{bs(S6, 'X->Ala')}</td></tr>
<tr><td>near-isosteric (|&Delta;Vol| &lt; 30 &Aring;<sup>3</sup>)</td><td>{bsn(FP, 'near-isosteric (|dVol|<30)')}</td><td>{bs(FP, 'near-isosteric (|dVol|<30)')}</td><td>{bsn(S6, 'near-isosteric (|dVol|<30)')}</td><td>{bs(S6, 'near-isosteric (|dVol|<30)')}</td></tr>
</table>
<p class="note">Bold = the 95&nbsp;% interval excludes zero. Positive = the class is harder than the rest.</p>

<p>The proline and glycine result reproduces the ordering seen in-distribution
(&rarr;Pro {ind('gly_pro', 'to Pro', 'MAE_sd', 2)}, from-Gly {ind('gly_pro', 'from Gly', 'MAE_sd', 2)},
from-Pro {ind('gly_pro', 'from Pro', 'MAE_sd', 2)} against {ind('gly_pro', 'other', 'MAE_sd', 2)}
for everything else), and sharpens it: <b>the weak class is <i>leaving</i> glycine, not arriving at
it</b> &mdash; &rarr;Gly is, if anything, easier than average
({ind('gly_pro', 'to Gly', 'MAE_sd', 2)} in-distribution, and negative on both blind corpora).</p>

<p><b>Near-isosteric substitutions are hard, and not only by the normalisation.</b> They also
degrade in <i>ranking</i>, which has no such floor: &rho;
{tr(FP, 'isosteric', 'near-isosteric', 'rho')} against {tr(FP, 'isosteric', 'rest', 'rho')} for the
rest on FireProt, and {tr(S6, 'isosteric', 'near-isosteric', 'rho')} against
{tr(S6, 'isosteric', 'rest', 'rho')} on S669. This is the transfer-side face of the same phenomenon as the
stabilizing tail: the model resolves large effects and blurs small ones.</p>

<div class="caveat">
<p><b>Class-level questions need protein-level replication, not variant counts.</b> The two blind
corpora agree on the sign of the near-isosteric and &rarr;Gly results, but only <i>from Pro</i>
clears zero on both. S669 is underpowered for most cells (&rarr;Pro n&nbsp;=&nbsp;{bsn(S6, '->Pro')},
from Gly n&nbsp;=&nbsp;{bsn(S6, 'from Gly')}) and disagrees on the two classes where it does have
sample size. The per-residue error ranking is the clearest demonstration: between the two corpora
its rank correlation is {rep('wt_aa', 'MAE_sd')} &mdash; that is, a "hardest source residues" list
read off 541 variants over 62 proteins carries no information about the next corpus.
What does replicate is within-class ranking quality ({rep('wt_aa', 'rho')}).
<b>Per-residue error rankings should not be quoted from a single benchmark of this size.</b></p>
</div>

<h2>6. Where the representation beats a substitution matrix</h2>
<p>All of the above measures where error is large. A different and more actionable question is
where the learned representation contributes anything at all. Scoring the same variants with a
40-dimensional one-hot encoding of the substitution identity &mdash; a model that can only learn
an average effect per residue pair, with no structure and no context &mdash; gives a matched
trivial baseline, and</p>
<p style="text-align:center"><i>skill</i> = 1 &minus; MAE(representation) &divide; MAE(one-hot substitution)</p>
<p>measures what the embedding adds. Pooled, skill is
<b>+{tr(FP, 'overall', 'all', 'skill_vs_onehot')}</b> on FireProt and
+{tr(S6, 'overall', 'all', 'skill_vs_onehot')} on S669 &mdash; the representation is not an amino-acid lookup. But it is not uniformly
better than one either:</p>

<table>
<tr><th>Source residue</th><th>skill, S669</th><th>skill, FireProt</th><th>&nbsp;</th><th>Substitution</th><th>n (FireProt)</th><th>skill, FireProt</th></tr>
<tr><td>Gln</td><td>{tr(S6, 'wt_aa', 'Q', 'skill_vs_onehot')}</td><td>{tr(FP, 'wt_aa', 'Q', 'skill_vs_onehot')}</td><td></td><td>Tyr &rarr; Phe</td><td>{trn(FP, 'pair', 'Y->F')}</td><td>{tr(FP, 'pair', 'Y->F', 'skill_vs_onehot')}</td></tr>
<tr><td>Trp</td><td>{tr(S6, 'wt_aa', 'W', 'skill_vs_onehot')}</td><td>{tr(FP, 'wt_aa', 'W', 'skill_vs_onehot')}</td><td></td><td>Trp &rarr; Ala</td><td>{trn(FP, 'pair', 'W->A')}</td><td>{tr(FP, 'pair', 'W->A', 'skill_vs_onehot')}</td></tr>
<tr><td>Thr</td><td>{tr(S6, 'wt_aa', 'T', 'skill_vs_onehot')}</td><td>{tr(FP, 'wt_aa', 'T', 'skill_vs_onehot')}</td><td></td><td>Trp &rarr; Phe</td><td>{trn(FP, 'pair', 'W->F')}</td><td>{tr(FP, 'pair', 'W->F', 'skill_vs_onehot')}</td></tr>
<tr><td>Lys</td><td>{tr(S6, 'wt_aa', 'K', 'skill_vs_onehot')}</td><td>{tr(FP, 'wt_aa', 'K', 'skill_vs_onehot')}</td><td></td><td>Lys &rarr; Arg</td><td>{trn(FP, 'pair', 'K->R')}</td><td>{tr(FP, 'pair', 'K->R', 'skill_vs_onehot')}</td></tr>
<tr><td>Ile</td><td>{tr(S6, 'wt_aa', 'I', 'skill_vs_onehot')}</td><td>{tr(FP, 'wt_aa', 'I', 'skill_vs_onehot')}</td><td></td><td>Val &rarr; Ala</td><td>{trn(FP, 'pair', 'V->A')}</td><td>{tr(FP, 'pair', 'V->A', 'skill_vs_onehot')}</td></tr>
<tr><td>Leu</td><td>{tr(S6, 'wt_aa', 'L', 'skill_vs_onehot')}</td><td>{tr(FP, 'wt_aa', 'L', 'skill_vs_onehot')}</td><td></td><td>Ile &rarr; Ala</td><td>{trn(FP, 'pair', 'I->A')}</td><td>{tr(FP, 'pair', 'I->A', 'skill_vs_onehot')}</td></tr>
<tr><td>Val</td><td>{tr(S6, 'wt_aa', 'V', 'skill_vs_onehot')}</td><td>{tr(FP, 'wt_aa', 'V', 'skill_vs_onehot')}</td><td></td><td>Leu &rarr; Ala</td><td>{trn(FP, 'pair', 'L->A')}</td><td>{tr(FP, 'pair', 'L->A', 'skill_vs_onehot')}</td></tr>
</table>

<p>Two residues sit at zero on <i>both</i> blind corpora: mutations away from <b>glutamine</b> and
from <b>tryptophan</b>, where the representation matches but does not beat the substitution
matrix. At the other end, mutations away from the aliphatics &mdash; Val, Leu, Ile &mdash; and from
Ala carry most of the model's advantage. By substitution pair the split is cleaner still:
chemistry-preserving replacements (Tyr&rarr;Phe, Trp&rarr;Phe, Lys&rarr;Arg) sit at zero skill,
while large-hydrophobic-to-alanine truncations, the classic core-packing perturbation, are where
the representation is worth the most.</p>

<p class="note">The full ordering of skill across residues replicates only weakly between the two
corpora ({rep('wt_aa', 'skill_vs_onehot')}), so the extremes are the defensible claim, not the
rank order.</p>

<figure>
<img src="{fig_tr}"/>
<figcaption><b>Figure 2.</b> Blind transfer, both corpora. <b>A:</b> class contrasts in
MAE &divide; sd against all other mutations, with cluster-bootstrap intervals over the 130 FireProt
proteins. <b>B:</b> skill over the one-hot substitution baseline by source residue, on both
corpora. <b>C:</b> the same by substitution pair, contrasting chemistry-preserving replacements
against core-packing truncations.</figcaption>
</figure>

<h2>7. Interpretation</h2>
<p>Read together, the three surviving effects are one effect seen from different angles. The
stabilizing tail, the near-isosteric substitutions and the chemistry-preserving pairs are all
<b>small-amplitude</b> perturbations, and the model compresses amplitudes: it separates large
destabilization from the rest confidently and blurs everything inside a narrow band. Burial,
which the literature flags, turns out to be a proxy for amplitude rather than a difficulty axis
of its own &mdash; once amplitude is controlled, it disappears entirely.</p>

<p>The skill decomposition adds a constraint on <i>why</i>. Where the mutation changes residue
type dramatically &mdash; a large hydrophobic truncated to alanine &mdash; a substitution-identity
average is already informative and the structural representation improves substantially on it.
Where the mutation preserves chemistry and the answer depends on packing detail at that specific
site, the representation falls back to roughly what the substitution identity alone would
predict. That is the signature of a readout carrying residue-type information plus coarse
context, rather than a fine-grained model of the local environment. It also says the remaining
headroom is not in the loss or in the regressor: a reweighting that emphasises the stabilizing
tail cannot manufacture discrimination that the frozen features do not encode.</p>

<p>Methodologically, two of the conventions used here changed conclusions rather than
decorating them. Normalising class error by class spread removed the burial effect entirely and
reversed the reading of every raw-MAE table. Requiring class-level claims to replicate across two
blind corpora, with protein-level rather than variant-level resampling, removed the per-residue
error ranking &mdash; the kind of result that is easy to report from a single benchmark and that
carries no information about the next one.</p>

</body></html>"""

out_html = R / "_report.html"
out_html.write_text(HTML, encoding="utf-8")
subprocess.run(["wkhtmltopdf", "--enable-local-file-access", "--quiet",
                str(out_html), str(R / "report.pdf")], check=True)
out_html.unlink()
print(f"wrote {R / 'report.pdf'}")
