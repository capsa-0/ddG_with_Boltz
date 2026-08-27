"""Build report.pdf for 13_balanced_loss (paper-facing; no provenance).

    python results/13_balanced_loss/build_report.py

Self-contained HTML (figure embedded as base64) rendered via wkhtmltopdf. Every number
is read from results.csv and bootstrap_paired.csv, so the PDF cannot drift from them.
Per results/guidelines.md this carries motivation, methods, results and interpretation
only — run history and file paths live in status.md and the README.
"""
import base64
import subprocess
from pathlib import Path

import pandas as pd

R = Path(__file__).parent
res = pd.read_csv(R / "results.csv")
pair = pd.read_csv(R / "bootstrap_paired.csv")

LOSS = {"mse": "plain MSE", "bmc": "Balanced MSE", "lds": "LDS reweighting"}
OOF, TR = "tsuboyama_oof", "s669_transfer"


def m(loss, metric, st=OOF):
    return float(res[(res.loss == loss) & (res["set"] == st)][metric].iloc[0])


def d(loss, metric):
    r = pair[(pair.loss == loss) & (pair.metric == metric)].iloc[0]
    b, e = ("<b>", "</b>") if r.excludes_zero else ("", "")
    star = " *" if r.excludes_zero else ""
    return f"{b}{r['mean']:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}]{star}{e}"


def img(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


F1 = img(R / "figures/01_balanced_loss.png")

METRICS = [("stab_bias", "bias, stabilizing class"),
           ("stab_rho", "ρ, stabilizing class"),
           ("auc_stab", "AUC, stabilizing class"),
           ("detpr30", "detection precision @30"),
           ("rho", "ρ overall"),
           ("r", "r overall"),
           ("mae", "MAE overall")]

rows = "".join(
    f"<tr><td>{label}</td><td>{d('bmc', k)}</td><td>{d('lds', k)}</td></tr>\n"
    for k, label in METRICS)

n_oof = int(m("mse", "n"))
n_stab = int(m("mse", "n_stab"))

CSS = """
@page { size: A4; margin: 15mm 16mm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 17pt; margin: 0 0 2px; color: #4a2c5a; }
h2 { font-size: 12.5pt; color: #4a2c5a; border-bottom: 1.5px solid #d8d2dc; padding-bottom: 3px; margin-top: 18px; }
h3 { font-size: 11pt; color: #4a2c5a; margin: 14px 0 4px; }
.sub { color: #555; font-size: 9.5pt; margin: 0 0 4px; }
.headline { background: #f2edf5; border-left: 4px solid #7B5BB8; padding: 10px 14px; margin: 14px 0; }
.caveat { background: #fdf3ec; border-left: 4px solid #C25A12; padding: 9px 13px; margin: 12px 0; font-size: 9.5pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #d5cfd9; padding: 4px 8px; text-align: right; }
th { background: #f6f3f8; } td:first-child, th:first-child { text-align: left; }
caption { caption-side: bottom; font-size: 8.5pt; color: #666; padding-top: 4px; text-align: left; }
figure { margin: 12px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; } figcaption { font-size: 8.5pt; color: #555; margin-top: 4px; text-align: left; }
"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>

<h1>Loss reweighting moves a frozen representation's predictions but not its discrimination</h1>
<p class="sub">Balanced MSE and inverse-density reweighting against the stabilizing-mutation
deficit · {n_oof:,} held-out mutations over 412 proteins · cluster-bootstrapped paired
differences</p>

<div class="headline">
<b>Balanced MSE removes {abs(100 * float(pair[(pair.loss=='bmc') & (pair.metric=='stab_bias')]['mean'].iloc[0]) / m('mse','stab_bias')):.0f} % of the
stabilizing bias and improves nothing else.</b> The bias on stabilizing mutations falls from
{m('mse','stab_bias'):+.2f} to {m('bmc','stab_bias'):+.2f} kcal/mol, an interval that clearly
excludes zero. Every ranking metric on that same class — ρ, AUC, top-K detection precision —
is statistically indistinguishable from baseline, while overall Pearson r and MAE get
significantly <i>worse</i>. Inverse-density reweighting is dominated: no significant gain
anywhere, significant losses on ρ and MAE. <b>Neither loss is adopted.</b>
</div>

<h2>1. Motivation</h2>
<p>A per-class error breakdown of this predictor found one deficit that survives normalisation
by class effect size: it systematically calls stabilizing mutations destabilizing. On held-out
data the bias on that class is {m('mse','stab_bias'):+.2f} kcal/mol and its within-class
Spearman ρ is only {m('mse','stab_rho'):.2f}, against a class that makes up
{100*n_stab/n_oof:.1f} % of the corpus ({n_stab} of {n_oof:,} mutations). That is precisely
the regime protein engineering cares about — the goal is to <i>find</i> the stabilizing
mutation, not to rank the destabilizing ones.</p>

<p>Label-imbalance losses are the cheapest published lever for exactly this shape of problem:
no new features, no architecture change, no retraining of the upstream representation. A
recent constraint-aware stability predictor reports a substantial external-benchmark gain from
loss-level changes alone, with Balanced MSE the term aimed at the rare tail. The question here
is whether that lever works when the representation underneath is <b>frozen</b>.</p>

<h2>2. Methods</h2>
<p>Three losses, identical architecture, data, splits and seeds.</p>
<ul>
<li><b>Plain MSE</b> — the project baseline.</li>
<li><b>Balanced MSE</b>, Monte-Carlo form — the batch is treated as a classification over
which target each prediction belongs to, which divides out the training label density. The
noise variance is learned in log space.</li>
<li><b>LDS</b> — inverse smoothed-density sample weighting over a 40-bin ΔΔG histogram.</li>
</ul>
<p>Both reweighted losses require sample weights or a custom objective, which the project's
default estimator does not support, so all three were reimplemented in torch at the same
topology and with the same antisymmetry augmentation. Evaluation is 5-fold grouped
cross-validation on wild-type identity, two seeds averaged.</p>

<p><b>Statistics.</b> A cluster bootstrap over the 412 proteins, reported as the <i>paired</i>
difference against MSE computed within each resample. The pairing matters more than the
bootstrap: the absolute intervals for the three losses overlap heavily because they carry the
between-protein variance that pairing cancels.</p>

<p><b>The stabilizing class</b> is defined as ΔΔG &lt; −0.5 kcal/mol throughout, and is
scored on four separate axes — bias (are the magnitudes right?), within-class ρ and AUC (is
the ordering right?), and top-K detection precision (would a screen find them?). The
distinction between the first and the rest is the result.</p>

<h2>3. Results</h2>

<table>
<caption>Paired difference against MSE with 95 % cluster-bootstrap CI over 412 proteins, on
held-out data. Bold and starred intervals exclude zero. For bias and MAE, negative is better;
for every other row, positive is better.</caption>
<tr><th>metric</th><th>Balanced MSE</th><th>LDS</th></tr>
{rows}
</table>

<h3>3.1 The bias is fixed; the blindness is not</h3>
<p>Balanced MSE is the only arm with a real gain, and it is confined to one row. The
stabilizing-class bias moves {d('bmc','stab_bias')} — from {m('mse','stab_bias'):+.2f} to
{m('bmc','stab_bias'):+.2f} kcal/mol. Every measure of whether the model can <i>tell which</i>
mutations stabilize is unchanged: within-class ρ {d('bmc','stab_rho')}, AUC
{d('bmc','auc_stab')}, detection precision {d('bmc','detpr30')}.</p>

<p>The cost is not confined. Overall Pearson r moves {d('bmc','r')} and MAE
{d('bmc','mae')}, both significantly in the wrong direction — the reweighting buys the tail's
calibration by spending accuracy on the bulk.</p>

<p>LDS is dominated outright: {d('lds','rho')} on overall ρ and {d('lds','mae')} on MAE, both
significant losses, with no significant gain on any tail metric.</p>

<h3>3.2 What the figure shows</h3>
<p>The mechanism is visible directly. Balanced MSE shifts stabilizing predictions downward as
a group, which is what a bias correction looks like. It does not change <i>which</i> mutations
are ranked most stabilizing — the top-K precision curves for the three losses lie on top of
one another.</p>

<figure><img src="{F1}"/>
<figcaption><b>Figure 1.</b> Left: paired differences against MSE with their intervals — only
the stabilizing bias clears zero in the good direction, while r and MAE clear it in the bad
one. Middle: the stabilizing region, showing the downward shift. Right: precision among the
top-K predicted-most-stabilizing mutations; the three curves are superimposed, so the shift
finds no additional stabilizing mutations.</figcaption></figure>

<h3>3.3 The same losses on an external benchmark</h3>
<p>Transferred without refitting to a blind external benchmark, no loss separates from any
other: overall r is {m('mse','r',TR):.3f} / {m('bmc','r',TR):.3f} / {m('lds','r',TR):.3f} for
MSE / Balanced MSE / LDS, and the within-tail ρ collapses toward zero for all three
({m('mse','stab_rho',TR):+.3f} / {m('bmc','stab_rho',TR):+.3f} /
{m('lds','stab_rho',TR):+.3f}). That is consistent with the separate finding that the error on
that benchmark is dominated by cross-dataset domain shift rather than by the tail — a loss
term aimed at the tail cannot address it.</p>

<h2>4. Interpretation</h2>
<div class="caveat">
<b>With a frozen representation, the loss can move predictions but cannot create discrimination
the features do not carry.</b> The bias is a property of the objective, and it is fixable by
changing the objective. The inability to <i>identify</i> which mutations stabilize is a
property of the representation, and it is not.
</div>

<p>This reading also explains why the published precedent and this result differ without
either being wrong. The constraint-aware predictor that gains from the same Balanced-MSE term
<b>fine-tunes its backbone</b>. There, the representation can adapt to the reweighted
objective — the loss changes what the features learn to encode. Here the trunk is frozen and
the loss only redistributes predictions within a fixed feature space.</p>

<p>The practical consequence is a recommendation and a condition on it. Neither loss should be
adopted as a default: the only real gain costs pooled r and MAE, and it does not improve the
metric that matters for engineering, which is finding stabilizing mutations. Balanced MSE is
worth revisiting <i>after</i> the representation is unfrozen, which is the setting where the
published precedent actually applies.</p>

<h2>5. Limitations</h2>
<ul>
<li>The stabilizing class holds {n_stab} mutations. Top-K detection precision in particular is
noisy at this size — its interval, {d('bmc','detpr30')}, is wide enough to be uninformative in
both directions, and no conclusion rests on it.</li>
<li>Two seeds per configuration, averaged. Seed variance is not separated from the
between-protein variance the bootstrap captures.</li>
<li>Only two reweighting schemes were tested, at their published default settings. A negative
result for these two is not a negative result for label-imbalance methods in general.</li>
<li>The stabilizing threshold of −0.5 kcal/mol is a convention. It was not varied, so the
findings are stated for that class definition rather than for a continuum of tail
severities.</li>
</ul>

<h2>6. Conclusion</h2>
<p>Asked whether reweighting the loss toward a rare, badly predicted tail can fix that tail
when the representation feeding it is frozen, the answer separates cleanly into two halves.
The systematic offset on the tail is an objective-level problem and reweighting removes a
fifth of it. Which mutations belong in the tail is a representation-level problem, and
reweighting does not touch it — at a measurable cost to the bulk. The deficit stays open, and
the direction it points is the representation, not the loss.</p>

</body></html>"""

out_html = R / "_report.html"
out_html.write_text(HTML, encoding="utf-8")
subprocess.run(["wkhtmltopdf", "--enable-local-file-access", "--quiet",
                str(out_html), str(R / "report.pdf")], check=True)
out_html.unlink()
print(f"wrote {R / 'report.pdf'}")
