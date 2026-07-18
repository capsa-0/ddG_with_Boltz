"""Build report.pdf for 02_stress_extrapolation (paper-facing)."""
import base64
import json
import subprocess
from pathlib import Path

R = Path(__file__).parent
s = json.load(open(R / "extrapolation_summary.json"))
i, t = s["in_distribution"], s["tail"]
fig = "data:image/png;base64," + base64.b64encode(
    (R / "figures/01_extrapolation_pred_vs_actual.png").read_bytes()).decode()

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

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<h1>Extrapolation to the destabilizing tail</h1>
<p class="sub">Experiment 02 · ddG_with_Boltz · raw-Δz features · gradient-boosted trees.</p>
<div class="headline">
Trained only on mild mutations (|ΔΔG| &lt; 1 kcal/mol) and tested on the strongly
destabilizing tail (ΔΔG &gt; 2), the predictor <b>does not extrapolate</b>: on the tail the
correlation is ~0 and the predicted-vs-measured slope is ~0.02 — predictions saturate near
the top of the training range while true values reach 5.7 kcal/mol. The model interpolates
within the ΔΔG band it was trained on and cannot reach beyond it.
</div>
<h2>1. Motivation</h2>
<p>The generalization study noted a predicted-vs-measured slope below 1 — the model
under-predicts large effects (regression to the mean). This test pushes that to the extreme:
by withholding the entire destabilizing tail from training, it distinguishes genuine
extrapolation from interpolation within the training range.</p>
<h2>2. Methods</h2>
<p>A gradient-boosted-tree regressor on 256 raw-Δz features is fit on the mild set
(|ΔΔG| &lt; 1) and evaluated on (a) held-out mild mutations — the in-distribution baseline —
and (b) the destabilizing tail (ΔΔG &gt; 2). We report Pearson r, RMSE, MAE, the linear fit
slope (measured = a + b·predicted), and how much of the true range the predictions cover.</p>
<h2>3. Results</h2>
<table>
<caption>Table 1. In-distribution baseline vs the extrapolation tail.</caption>
<tr><th>Split</th><th>n</th><th>Pearson r</th><th>RMSE</th><th>MAE</th><th>fit slope</th><th>pred max / true max</th></tr>
<tr><td>In-distribution (held-out |ΔΔG|&lt;1)</td><td>{i['n']}</td><td>{i['pearson']:.3f}</td><td>{i['rmse']:.2f}</td><td>{i['mae']:.2f}</td><td>{i['slope']:.2f}</td><td>{i['pred_max']:.2f} / {i['true_max']:.2f}</td></tr>
<tr><td><b>Extrapolation tail (ΔΔG&gt;2)</b></td><td>{t['n']}</td><td><b>{t['pearson']:.3f}</b></td><td><b>{t['rmse']:.2f}</b></td><td>{t['mae']:.2f}</td><td><b>{t['slope']:.2f}</b></td><td><b>{t['pred_max']:.2f} / {t['true_max']:.2f}</b></td></tr>
</table>
<figure><img src="{fig}"><figcaption><b>Figure 1.</b> Predicted vs measured ΔΔG for the
in-distribution baseline and the tail, with y=x and fit lines. The tail is a flat cloud.</figcaption></figure>
<h2>4. Discussion</h2>
<p>On the tail the correlation collapses to {t['pearson']:.2f} and the fit slope to
{t['slope']:.2f} — essentially flat. Predictions cap at ~{t['pred_max']:.1f} kcal/mol, the top
of the training range, while true ΔΔG reaches {t['true_max']:.1f}. This is the
regression-to-the-mean weakness in its extreme form: the model interpolates within the ΔΔG
range it has seen and cannot extrapolate beyond it. The practical implication is direct — to
rank or screen strongly destabilizing mutations, the training set must itself span that range;
no amount of modeling recovers effect sizes absent from training.</p>
</body></html>"""

(R / "report.html").write_text(HTML)
subprocess.run(["wkhtmltopdf", "--enable-local-file-access", "--quiet",
                str(R / "report.html"), str(R / "report.pdf")], check=True)
(R / "report.html").unlink()
print("wrote", R / "report.pdf")
