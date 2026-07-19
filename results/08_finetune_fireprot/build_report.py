"""Build report.pdf for 08_finetune_fireprot (paper-facing; no provenance).

    python results/08_finetune_fireprot/build_report.py
"""
import base64
import subprocess
from pathlib import Path

import pandas as pd

R = Path(__file__).parent
df = pd.read_csv(R / "results.csv")


def v(thr, cond, test, col="pearson"):
    return float(df[(df.thr == thr) & (df.cond == cond) & (df.test == test)][col].iloc[0])


def img(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


fig = img(R / "figures/01_finetune_bars.png")

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
"""


def frow(thr):  # FireProt-test: A / B / D (Pearson, Spearman)
    return (f"<tr><td>{thr}%</td>"
            f"<td>{v(thr,'A_tsu_only','fp_test'):.3f}</td>"
            f"<td>{v(thr,'B_fp_only','fp_test'):.3f}</td>"
            f"<td><b>{v(thr,'D_finetuned','fp_test'):.3f}</b></td>"
            f"<td>{v(thr,'A_tsu_only','fp_test','spearman'):.3f} / "
            f"{v(thr,'B_fp_only','fp_test','spearman'):.3f} / "
            f"<b>{v(thr,'D_finetuned','fp_test','spearman'):.3f}</b></td></tr>")


def trow(thr):  # Tsuboyama-test: A / B / D (Pearson)
    return (f"<tr><td>{thr}%</td>"
            f"<td>{v(thr,'A_tsu_only','tsu_test'):.3f}</td>"
            f"<td>{v(thr,'B_fp_only','tsu_test'):.3f}</td>"
            f"<td>{v(thr,'D_finetuned','tsu_test'):.3f}</td></tr>")


HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<h1>Fine-tuning a Tsuboyama-pretrained ΔΔG model on FireProt</h1>
<p class="sub">Experiment 08 · ddG_with_Boltz · concat features + antisymmetry · MLP · cross-dataset homology holdout.</p>

<div class="headline">
On a FireProt ≤500 test set of 25–27 homology-held-out proteins, <b>sequentially fine-tuning</b>
a Tsuboyama-pretrained ΔΔG regressor on FireProt <b>does not reliably improve FireProt accuracy</b>:
plain Tsuboyama-only transfer is the best in Pearson at the 30 % and 50 % thresholds, fine-tuning
wins only at 90 %, and in rank correlation the two are within noise. The one robust effect is that
training on FireProt <b>alone</b> forgets Tsuboyama. The winning recipe is therefore big-corpus
pretraining + transfer, not fine-tuning on the small target set — consistent with the published
literature and with this project's density-limited picture of the error.
</div>

<h2>1. Motivation</h2>
<p>A Tsuboyama-trained model transfers to the independently curated FireProt dataset but is
bounded by how densely the training data covers each ΔΔG value. A natural way to add coverage
is to fine-tune on FireProt itself. The question is whether this helps FireProt, and whether it
degrades the original Tsuboyama performance (catastrophic forgetting) — measured on both
datasets under a split that controls for homology across them.</p>

<h2>2. Methods</h2>
<p><b>Splits.</b> All wild-type sequences from both datasets are pooled and clustered by
sequence identity (MMseqs2, 80% coverage) at 30/50/90%. Whole clusters are assigned to train
or test, so no train/test pair — within or across datasets — exceeds the threshold; clusters
spanning both datasets go to train. This yields disjoint <i>Tsuboyama-train / Tsuboyama-test /
FireProt-finetune / FireProt-test</i> sets.</p>
<p><b>Model.</b> 5-seed MLP ensemble on concat pair-track features with antisymmetry
augmentation (the defaults established in experiment 07), evaluated in three conditions.
<b>A (Tsuboyama-only):</b> trained on Tsuboyama-train. <b>B (FireProt-only):</b> a fresh model
trained on FireProt-finetune alone, with its own input normalisation — the baseline for "how far
does FireProt get on its own." <b>D (fine-tuned):</b> the pretrained model, warm-start continued
on FireProt-finetune at a lower learning rate, reusing the Tsuboyama normalisation. All three are
evaluated on Tsuboyama-test and FireProt-test.</p>

<h2>3. Results</h2>
<table>
<caption>Table 1. FireProt-test pooled correlation, A / B / D. Bold = best (D).</caption>
<tr><th>Identity</th><th>Pearson A</th><th>Pearson B</th><th>Pearson D</th><th>Spearman A / B / D</th></tr>
{frow(30)}{frow(50)}{frow(90)}
</table>
<table>
<caption>Table 2. Tsuboyama-test pooled Pearson r, A / B / D (forgetting check).</caption>
<tr><th>Identity</th><th>A (Tsu-only)</th><th>B (FP-only)</th><th>D (fine-tuned)</th></tr>
{trow(30)}{trow(50)}{trow(90)}
</table>

<figure><img src="{fig}">
<figcaption><b>Figure 1.</b> Pooled Pearson r for A / B / D. Left: FireProt-test — Tsuboyama-only
(A) is best at 30/50 %, fine-tuning (D) only at 90 %. Right: Tsuboyama-test — FireProt-only (B)
drops well below A/D (forgets); the 90 % A/D bars are a calibration outlier on that split.</figcaption></figure>

<h2>4. Discussion</h2>
<p>On this ≤500 test set — 25–27 homology-held-out proteins, ~2× the earlier ≤200 set —
<b>fine-tuning does not reliably beat plain transfer</b>. In Pearson, Tsuboyama-only (A) is the best
FireProt-test model at 30 % and 50 % identity; fine-tuning (D) wins only at 90 %, and in Spearman
the two are within noise (D marginally ahead at all three). An earlier, smaller ≤200 experiment
had suggested a modest fine-tuning gain; on the larger, less noisy test set that gain washes out,
which is the more trustworthy reading. The result is consistent with the published literature
(ThermoMPNN reports that fine-tuning on FireProt does not reliably help and that training on
FireProt alone degrades). The one robust effect here is exactly that: the <b>FireProt-only baseline
(B) forgets Tsuboyama</b> — its Tsuboyama-test correlation drops from ~0.80 to ~0.66–0.72 —
because training on the small target set alone discards the broad Tsuboyama signal. The practical
conclusion is that the predictor's accuracy is set by the pair-track features and the coverage of
the large pretraining corpus, not by exposure to FireProt labels; big-corpus pretraining followed
by transfer is the recipe, and fine-tuning on the small target set adds little.</p>
</body></html>"""

(R / "report.html").write_text(HTML)
subprocess.run(["wkhtmltopdf", "--enable-local-file-access", "--quiet",
                str(R / "report.html"), str(R / "report.pdf")], check=True)
(R / "report.html").unlink()
print("wrote", R / "report.pdf")
