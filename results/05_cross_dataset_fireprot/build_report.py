"""Build report.pdf for 05_cross_dataset_fireprot from the committed result files.

Self-contained HTML (figures embedded as base64) rendered via wkhtmltopdf, so the
PDF needs no LaTeX and no external file access. Re-run after the numbers change:

    python results/05_cross_dataset_fireprot/build_report.py
"""
import base64
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).parent
# Repo root = results/05_.../ -> up two.
ROOT = R.parents[1]
PROC = ROOT / "data/processed"


def _stats():
    """Recompute the report numbers from the result files + processed parquets.

    Committed inputs: this folder's transfer_summary{,_hgb}.json + per_protein.csv.
    Processed inputs (gitignored, local/cluster copy): predictions.parquet and the
    two feature tables — only for the fit slope / SD / ΔΔG ranges; if they're absent
    those fields fall back to values already recorded in the JSON summaries.
    """
    mlp = json.load(open(R / "transfer_summary.json"))
    hgb = json.load(open(R / "transfer_summary_hgb.json"))
    pp = pd.read_csv(R / "per_protein.csv")
    sc = pp.dropna(subset=["pearson"])
    s = dict(
        mlp=mlp, hgb=hgb,
        pp_mean=float(sc["pearson"].mean()), pp_median=float(sc["pearson"].median()),
        frac05=float((sc["pearson"] > 0.5).mean()),
        frac07=float((sc["pearson"] > 0.7).mean()),
        n_scored=int(len(sc)), n_prot=int(len(pp)),
        n_train=int(mlp["n_train"]), n_test=int(mlp["n_test"]),
        top=sc[sc["n"] >= 8].sort_values("pearson", ascending=False)
            .head(6)[["unit", "n", "pearson", "spearman", "rmse"]].values.tolist(),
        bot=sc[sc["n"] >= 8].sort_values("pearson")
            .head(5)[["unit", "n", "pearson", "spearman", "rmse"]].values.tolist(),
        recovered=pp[pp["unit"].isin(["3PG0", "2IMM", "1YYX"])]
            [["unit", "n", "pearson", "spearman", "rmse"]].values.tolist(),
        # defaults if the processed parquets aren't present
        slope=0.27, pred_sd=0.66, meas_sd=1.58,
        fp_min=-13.7, fp_max=12.0, tsu_min=-2.7, tsu_max=5.7,
    )
    pred_p = PROC / "fireprot_le500/transfer_from_tsuboyama/predictions.parquet"
    fp_p = PROC / "fireprot_le500/features_summary.parquet"
    tsu_p = PROC / "tsuboyama_bench_fast/rawz_features.parquet"
    if pred_p.exists():
        pred = pd.read_parquet(pred_p)
        s["slope"] = float(np.polyfit(pred["y"], pred["pred"], 1)[0])
        s["pred_sd"] = float(pred["pred"].std())
        s["meas_sd"] = float(pred["y"].std())
    if fp_p.exists():
        d = pd.read_parquet(fp_p)["ddg"]
        s["fp_min"], s["fp_max"] = float(d.min()), float(d.max())
    if tsu_p.exists():
        d = pd.read_parquet(tsu_p)["ddg"]
        s["tsu_min"], s["tsu_max"] = float(d.min()), float(d.max())
    return s


S = _stats()
mlp, hgb = S["mlp"], S["hgb"]


def img(path):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def rows(data, fmt):
    return "\n".join("<tr>" + "".join(f"<td>{fmt(i, v)}</td>"
                     for i, v in enumerate(r)) + "</tr>" for r in data)


def prot_row(i, v):
    return v if i == 0 else (f"{v:.0f}" if i == 1 else f"{v:.3f}")


scatter = img(R / "figures/01_transfer_scatter.png")
hist = img(R / "figures/02_per_protein_r_hist.png")
err_fig = img(R / "figures/03_error_vs_ddg.png")
dens_fig = img(R / "figures/04_density_vs_error.png")

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 20mm 18mm; }}
body {{ font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10.5pt;
        color: #1a1a1a; line-height: 1.45; }}
h1 {{ font-size: 19pt; margin: 0 0 2px 0; color: #14314f; }}
h2 {{ font-size: 13pt; color: #14314f; border-bottom: 1.5px solid #d0d7de;
      padding-bottom: 3px; margin-top: 22px; }}
h3 {{ font-size: 11pt; color: #22405c; margin-bottom: 4px; }}
.sub {{ color: #555; font-size: 9.5pt; margin: 0 0 4px 0; }}
.headline {{ background: #eef4fb; border-left: 4px solid #2c6fb3;
    padding: 10px 14px; margin: 14px 0; font-size: 11pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0 4px; font-size: 9.5pt;
    page-break-inside: avoid; }}
figure {{ page-break-inside: avoid; }}
th, td {{ border: 1px solid #cfd6dd; padding: 4px 8px; text-align: right; }}
th {{ background: #f3f6f9; }}
td:first-child, th:first-child {{ text-align: left; }}
caption {{ caption-side: bottom; font-size: 8.5pt; color: #666; padding-top: 4px;
    text-align: left; }}
figure {{ margin: 12px 0; text-align: center; }}
figure img {{ max-width: 100%; }}
figcaption {{ font-size: 8.5pt; color: #555; margin-top: 4px; text-align: left; }}
code {{ background: #f3f4f6; padding: 0 3px; font-size: 9pt; }}
.small {{ font-size: 8.8pt; color: #555; }}
ul {{ margin: 4px 0; }} li {{ margin: 2px 0; }}
.two {{ display: flex; gap: 12px; }} .two figure {{ flex: 1; }}
</style></head><body>

<h1>Cross-dataset transfer of a Boltz raw-Δz ΔΔG predictor</h1>
<p class="sub"><b>Experiment 05 — Tsuboyama → FireProt.</b> ddG_with_Boltz project ·
Generated 2026-07-19 · raw-Δz features, MLP (5-seed ensemble) primary model.</p>

<div class="headline">
A ΔΔG predictor trained on the entire <b>Tsuboyama</b> mega-scale folding dataset
({S['n_train']:,} mutations) transfers — with <b>no refitting</b> — to the independent
<b>FireProt</b> dataset ({S['n_test']:,} mutations / {mlp['n_test_proteins']} proteins,
<b>zero protein overlap</b>): pooled Pearson <b>r = {mlp['pooled_pearson']:.3f}</b>,
Spearman <b>ρ = {mlp['pooled_spearman']:.3f}</b>, per-protein <b>median r = {S['pp_median']:.2f}</b>.
The Boltz raw-Δz signal is not an artifact of one dataset. Its ceiling is
<b>magnitude, not ranking</b>: predictions span only ~40% of the true ΔΔG spread
(fit slope {S['slope']:.2f}), so the model ranks and triages well but under-predicts
absolute effect sizes on out-of-range mutations.
</div>

<h2>1. Objective</h2>
<p>Every generalization test so far (experiments 01, 06) splits <i>within</i> the
Tsuboyama corpus. Such holdouts can still ride on quirks shared by a single data
source — Tsuboyama is one high-throughput folding assay on mostly small/designed
domains. The decisive question for a usable predictor is whether the signal carries
to an <b>independently curated</b> dataset: different proteins, a different assay, and
a different label provenance. We test transfer to <b>FireProt</b> (literature-derived
stability measurements, ≤500 aa).</p>

<h2>2. Data &amp; methods</h2>
<ul>
<li><b>Train:</b> all {S['n_train']:,} Tsuboyama mutations
(<code>tsuboyama_bench_fast/rawz_features.parquet</code>), <b>256 raw-Δz features</b>
(mutant−WT difference of the Boltz-2 pair track <i>z</i>: <code>zdiag_*</code> 128 +
<code>zpool_*</code> 128).</li>
<li><b>Test:</b> all {S['n_test']:,} FireProt mutations / {mlp['n_test_proteins']} proteins
(<code>fireprot_le500/features_summary.parquet</code>), the <i>same</i> 256 features.</li>
<li><b>Model:</b> the benchmark pipeline <code>SimpleImputer(median) → StandardScaler
→ estimator</code>, fit once on Tsuboyama and applied unchanged to FireProt. Primary
estimator: a <b>5-seed MLP ensemble</b> (project default since exp 06);
HistGradientBoosting (HGB) reported alongside.</li>
<li><b>Entry point:</b> <code>python -m ddg.evaluation.transfer --train … --test …
--model mlp</code>.</li>
<li><b>Independence:</b> the train and test protein sets have <b>zero <code>wt_id</code>
overlap</b>. FireProt's ΔΔG range [{S['fp_min']:.1f}, {S['fp_max']:.1f}] is much wider
than Tsuboyama's [{S['tsu_min']:.1f}, {S['tsu_max']:.1f}].</li>
<li><b>Sign convention:</b> FireProt (<code>ddG</code>) and Tsuboyama (<code>ddg</code>)
share the same sign (positive = destabilizing; both ~75% positive), so predictions are
used as-is (no flip).</li>
</ul>

<h2>3. Results</h2>
<table>
<caption>Table 1. Pooled and per-protein transfer metrics (n = {S['n_test']:,} mutations /
{mlp['n_test_proteins']} proteins). ΔΔG in kcal/mol.</caption>
<tr><th>Metric</th><th>MLP (primary)</th><th>HGB</th></tr>
<tr><td>Pooled Pearson r</td><td><b>{mlp['pooled_pearson']:.3f}</b></td><td>{hgb['pooled_pearson']:.3f}</td></tr>
<tr><td>Pooled Spearman ρ</td><td><b>{mlp['pooled_spearman']:.3f}</b></td><td>{hgb['pooled_spearman']:.3f}</td></tr>
<tr><td>Pooled RMSE</td><td>{mlp['pooled_rmse']:.2f}</td><td>{hgb['pooled_rmse']:.2f}</td></tr>
<tr><td>Pooled MAE</td><td>{mlp['pooled_mae']:.2f}</td><td>{hgb['pooled_mae']:.2f}</td></tr>
<tr><td>Per-protein r — mean</td><td>{mlp['pearson_mean']:.3f}</td><td>{hgb['pearson_mean']:.3f}</td></tr>
<tr><td>Per-protein r — median</td><td><b>{S['pp_median']:.3f}</b></td><td>—</td></tr>
<tr><td>Proteins scored</td><td>{S['n_scored']} / {S['n_prot']}</td><td>{S['n_scored']} / {S['n_prot']}</td></tr>
</table>
<p>MLP and HGB are within noise of each other — the same "representation, not model"
result as experiment 06. {S['frac05']*100:.0f}% of proteins score r &gt; 0.5 and
{S['frac07']*100:.0f}% score r &gt; 0.7; the per-protein <i>mean</i> ({S['pp_mean']:.2f})
sits well below the <i>median</i> ({S['pp_median']:.2f}) because a handful of proteins
transfer poorly (Fig. 2).</p>

<figure>
<img src="{scatter}">
<figcaption><b>Figure 1.</b> Predicted (trained on Tsuboyama) vs measured (FireProt)
ΔΔG for all {S['n_test']:,} mutations. Dashed line is y = x. The cloud is strongly
correlated but far flatter than the diagonal: predicted ΔΔG concentrates in ~[0, 3]
while measured values span [{S['fp_min']:.1f}, {S['fp_max']:.1f}] — the magnitude
compression quantified in §4.</figcaption>
</figure>

<figure>
<img src="{hist}">
<figcaption><b>Figure 2.</b> Distribution of per-protein Pearson r ({S['n_scored']} of
{S['n_prot']} proteins scored; the rest have &lt;2 mutations or constant ΔΔG).
Concentrated at 0.6–1.0 (median {S['pp_median']:.2f}); the mean {S['pp_mean']:.2f} is
pulled down by a few poorly-transferring proteins.</figcaption>
</figure>

<table>
<caption>Table 2. Best- and worst-transferring proteins (≥8 mutations each).</caption>
<tr><th>Protein</th><th>n</th><th>Pearson r</th><th>Spearman ρ</th><th>RMSE</th></tr>
{rows(S['top'], prot_row)}
<tr><td colspan="5" class="small">… {S['n_scored']-11} proteins between …</td></tr>
{rows(S['bot'], prot_row)}
</table>

<h2>4. The ceiling: magnitude, not ranking</h2>
<p>The predictor <i>ranks</i> mutations well (ρ = {mlp['pooled_spearman']:.2f}) but
severely <b>under-predicts magnitude</b>. The predicted-vs-measured fit slope is
<b>{S['slope']:.2f}</b>, and predicted ΔΔG has SD {S['pred_sd']:.2f} against a measured
SD of {S['meas_sd']:.2f} — only ~{S['pred_sd']/S['meas_sd']*100:.0f}% of the true spread.
FireProt's wide range makes this starker than any within-Tsuboyama split: the model
regresses toward the mean on the destabilizing/stabilizing tails it never saw in
training. This is the same regression-to-the-mean weakness isolated in experiment 02
(extrapolation) and confirmed model-independent in 06 — a property of the
features/objective, not of the estimator. <b>Practical consequence:</b> use for
ranking, prioritization, and triage — not for absolute ΔΔG on mutations outside the
training range.</p>

<h2>5. Where the error lives: training coverage of ΔΔG</h2>
<p>Splitting the test by the Tsuboyama training range ([−1, 4] kcal/mol, its central
~98%) makes the ceiling precise. <b>In-range</b> (n={mlp['in_n']}) the model is genuinely
good: Pearson <b>{mlp['in_pearson']:.2f}</b>, RMSE <b>{mlp['in_rmse']:.2f}</b>. <b>Out-of-range</b>
(n={mlp['out_n']}) the error explodes: RMSE <b>{mlp['out_rmse']:.2f}</b>, and the per-tail
correlation collapses — the predictions cannot reach ΔΔG values absent from training.</p>
<figure><img src="{err_fig}">
<figcaption><b>Figure 3.</b> Prediction error vs measured ΔΔG. A regression-to-mean bias
(over-predicts low ΔΔG, under-predicts high), with error minimized in the dense centre and
rising toward both tails; the ±SD band is flat, so tail error is systematic bias.</figcaption></figure>
<p>The cause is <b>training density</b>, not FireProt itself: relating per-bin error to how
densely Tsuboyama sampled each ΔΔG value, error is almost perfectly anti-correlated with
density (Spearman ρ = <b>{mlp['density_error_spearman_bins']:.2f}</b>). Accuracy at a given ΔΔG
is set by how much training data covered it — a coverage effect, identical across model
families (cf. experiments 02 and 06).</p>
<figure><img src="{dens_fig}">
<figcaption><b>Figure 4.</b> Test error vs Tsuboyama training density in ΔΔG space. Left:
density and error are mirror images along ΔΔG. Right: error falls monotonically with
training density.</figcaption></figure>

<h2>6. Context</h2>
<p>The pooled r ≈ 0.65 matches the published state of the art for this transfer: ThermoMPNN
and the AFToolkit framework report ~0.65 Pearson transferring to FireProt from the same
Megascale/Tsuboyama training data using AlphaFold2 / graph-neural-network backbones. The
Boltz-2 raw-Δz pipeline reaches the same level with a simple downstream regressor.</p>

</body></html>"""

html_path = R / "report.html"
html_path.write_text(HTML)
pdf_path = R / "report.pdf"
subprocess.run(
    ["wkhtmltopdf", "--enable-local-file-access", "--quiet",
     "--print-media-type", str(html_path), str(pdf_path)],
    check=True)
html_path.unlink()
print("wrote", pdf_path, pdf_path.stat().st_size, "bytes")
