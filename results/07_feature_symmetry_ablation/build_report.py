"""Build report.pdf for 07_feature_symmetry_ablation (paper-facing; no provenance).

    python results/07_feature_symmetry_ablation/build_report.py
Self-contained HTML (figure embedded as base64) rendered via wkhtmltopdf.
"""
import base64
import subprocess
from pathlib import Path

import pandas as pd

R = Path(__file__).parent
df = pd.read_csv(R / "results.csv")


def g(ds, feat, aug, col="pearson"):
    return float(df[(df.dataset == ds) & (df.features == feat) & (df.augment == aug)][col].iloc[0])


def img(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


fig = img(R / "figures/01_ablation_bars.png")

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10.5pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 18pt; margin: 0 0 2px; color: #14314f; }
h2 { font-size: 13pt; color: #14314f; border-bottom: 1.5px solid #d0d7de; padding-bottom: 3px; margin-top: 20px; }
.sub { color: #555; font-size: 9.5pt; margin: 0 0 4px; }
.headline { background: #eef4fb; border-left: 4px solid #2c6fb3; padding: 10px 14px; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #cfd6dd; padding: 4px 8px; text-align: right; }
th { background: #f3f6f9; } td:first-child, th:first-child { text-align: left; }
caption { caption-side: bottom; font-size: 8.5pt; color: #666; padding-top: 4px; text-align: left; }
figure { margin: 12px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; } figcaption { font-size: 8.5pt; color: #555; margin-top: 4px; text-align: left; }
code { background: #f3f4f6; padding: 0 3px; font-size: 9pt; }
"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<h1>Feature form and antisymmetry augmentation for ΔΔG prediction</h1>
<p class="sub">Experiment 07 · ddG_with_Boltz · raw pair-track (z) features · MLP, protein-holdout.</p>

<div class="headline">
Two design choices for a ΔΔG regressor built on Boltz pair-track embeddings are settled by a
within-dataset ablation (Tsuboyama and FireProt, protein-holdout). <b>Concatenating the
wild-type and mutant pooled pair-vectors</b> (rather than only their difference) is
consistently as good or better, and <b>antisymmetry augmentation</b> (ΔΔG(A→B) = −ΔΔG(B→A))
improves FireProt — but only in the concat representation, where the reverse mutation is a
natural input transform. The two together are adopted as the default.
</div>

<h2>1. Motivation</h2>
<p>A ΔΔG predictor on Boltz-2 pair-track (<i>z</i>) embeddings can be built from the mutated
residue's pooled pair-vector in two ways: the <b>difference</b> of mutant and wild-type
(<code>Δz</code>), or their <b>concatenation</b> (<code>concat = [wt, mut]</code>, which keeps
both absolute levels; the difference is recoverable as their subtraction but not vice-versa).
Separately, ΔΔG obeys an exact antisymmetry — the reverse mutation has the negated value —
which can be injected as a training augmentation. We ask, on each dataset independently,
whether either helps.</p>

<h2>2. Methods</h2>
<p>For a mutation at residue <i>i</i>, the pooled pair-vector is the mean over partners of the
<i>z</i> row at <i>i</i> (128-dim), taken for the wild-type and mutant structures.
<b>Δz</b> = the difference (with the diagonal element); <b>concat</b> = the two pooled vectors
side by side. Both are 256-dim. <b>Antisymmetry augmentation</b> adds each reversed mutation
(negated ΔΔG) to the training set — for Δz by negating the feature vector, for concat by
swapping the two halves. Evaluation is a protein-holdout (5-fold GroupKFold by protein),
out-of-fold pooled metrics, with a 5-seed MLP ensemble. Corpora: Tsuboyama (12,359 mutations
/ 412 proteins) and FireProt (1,543 / 85).</p>

<h2>3. Results</h2>
<table>
<caption>Table 1. Pooled Pearson r (protein-holdout). Best per dataset in bold.</caption>
<tr><th>Dataset</th><th>Feature</th><th>no augmentation</th><th>+ antisymmetry</th></tr>
<tr><td>Tsuboyama</td><td>Δz (difference)</td><td>{g('Tsuboyama','dz','none'):.3f}</td><td>{g('Tsuboyama','dz','sym'):.3f}</td></tr>
<tr><td>Tsuboyama</td><td>concat</td><td><b>{g('Tsuboyama','concat','none'):.3f}</b></td><td>{g('Tsuboyama','concat','sym'):.3f}</td></tr>
<tr><td>FireProt</td><td>Δz (difference)</td><td>{g('FireProt','dz','none'):.3f}</td><td>{g('FireProt','dz','sym'):.3f}</td></tr>
<tr><td>FireProt</td><td>concat</td><td>{g('FireProt','concat','none'):.3f}</td><td><b>{g('FireProt','concat','sym'):.3f}</b></td></tr>
</table>

<figure><img src="{fig}">
<figcaption><b>Figure 1.</b> Protein-holdout Pearson r for the 2×2 ablation, per dataset.</figcaption></figure>

<h2>4. Discussion</h2>
<p><b>Concat ≥ Δz everywhere.</b> Keeping both wild-type and mutant levels is consistently at
least as good as the difference (Tsuboyama {g('Tsuboyama','dz','none'):.3f}→{g('Tsuboyama','concat','none'):.3f},
FireProt {g('FireProt','dz','none'):.3f}→{g('FireProt','concat','none'):.3f}) — a free gain,
since concat strictly contains the difference.</p>
<p><b>Antisymmetry augmentation is representation-dependent.</b> On Δz it forces an
odd-function model (f(−x) = −f(x)) that fights a dataset skewed toward destabilizing mutations,
collapsing Tsuboyama's calibration (Pearson {g('Tsuboyama','dz','none'):.3f}→{g('Tsuboyama','dz','sym'):.3f})
while rank correlation is preserved. In the concat representation the reverse mutation is
instead a swap of the two halves — a valid input point — so augmentation is safe on Tsuboyama
({g('Tsuboyama','concat','none'):.3f}→{g('Tsuboyama','concat','sym'):.3f}) and <b>improves
FireProt</b> ({g('FireProt','concat','none'):.3f}→{g('FireProt','concat','sym'):.3f}).</p>
<p><b>Conclusion.</b> Use <b>concat features with antisymmetry augmentation</b>: neutral on
Tsuboyama, a real gain on FireProt (+{g('FireProt','concat','sym')-g('FireProt','dz','none'):.3f}
Pearson over the plain-difference baseline).</p>
</body></html>"""

(R / "report.html").write_text(HTML)
subprocess.run(["wkhtmltopdf", "--enable-local-file-access", "--quiet",
                str(R / "report.html"), str(R / "report.pdf")], check=True)
(R / "report.html").unlink()
print("wrote", R / "report.pdf")
