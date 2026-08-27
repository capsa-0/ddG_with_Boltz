"""Build report.pdf for 09_external_benchmarks (paper-facing; no provenance).

    python results/09_external_benchmarks/build_report.py

Self-contained HTML (figures embedded as base64) rendered via wkhtmltopdf. Every
number is read from results.csv / results_pre-correction.csv / ssym_antisymmetry.csv,
so the PDF cannot drift from them. Per results/guidelines.md this carries motivation,
methods, results and interpretation only — dataset assembly, file paths and run
history live in status.md and the README provenance table.
"""
import base64
import subprocess
from pathlib import Path

import pandas as pd

R = Path(__file__).parent
res = pd.read_csv(R / "results.csv")
pre = pd.read_csv(R / "results_pre-correction.csv")
anti = pd.read_csv(R / "ssym_antisymmetry.csv")

REGS = ("A_tsu_only", "B_fp_only", "D_finetuned")
NAME = {"A_tsu_only": "A — Tsuboyama only", "B_fp_only": "B — FireProt only",
        "D_finetuned": "D — fine-tuned"}


def v(bench, reg, sub, col="pearson", table=res):
    row = table[(table.benchmark == bench) & (table.regime == reg) & (table.subset == sub)]
    return float(row[col].iloc[0])


def n(bench, reg, sub):
    return int(v(bench, reg, sub, "n"))


def ab(reg, col):
    return float(anti[anti.regime == reg][col].iloc[0])


def img(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


F1 = img(R / "figures/01_pooled_r_full_vs_filtered.png")
F2 = img(R / "figures/02_correction_and_antisymmetry.png")


def bench_rows(bench, subs):
    out = ""
    for reg in REGS:
        cells = "".join(
            f"<td>{v(bench, reg, s):.3f} <span class='pp'>({v(bench, reg, s, 'per_prot_median_r'):.2f})</span></td>"
            for s in subs)
        out += f"<tr><td>{NAME[reg]}</td>{cells}</tr>\n"
    return out


CSS = """
@page { size: A4; margin: 15mm 16mm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 17pt; margin: 0 0 2px; color: #123a52; }
h2 { font-size: 12.5pt; color: #123a52; border-bottom: 1.5px solid #d0d7da; padding-bottom: 3px; margin-top: 18px; }
h3 { font-size: 11pt; color: #123a52; margin: 14px 0 4px; }
.sub { color: #555; font-size: 9.5pt; margin: 0 0 4px; }
.headline { background: #eaf1f6; border-left: 4px solid #4C72B0; padding: 10px 14px; margin: 14px 0; }
.caveat { background: #fdf1f0; border-left: 4px solid #C44E52; padding: 9px 13px; margin: 12px 0; font-size: 9.5pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #cfd6da; padding: 4px 8px; text-align: right; }
th { background: #f2f5f7; } td:first-child, th:first-child { text-align: left; }
.pp { color: #777; font-size: 8pt; }
caption { caption-side: bottom; font-size: 8.5pt; color: #666; padding-top: 4px; text-align: left; }
figure { margin: 12px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; } figcaption { font-size: 8.5pt; color: #555; margin-top: 4px; text-align: left; }
"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>

<h1>Blind external benchmarks under homology control</h1>
<p class="sub">A Boltz-2 embedding ΔΔG predictor on S669 and Ssym · three training regimes ·
MMseqs2 leakage control at 25 % and 30 % identity</p>

<div class="headline">
<b>S669 is the honest hard test, and most of what looked like a training-corpus effect was
leakage or a defective estimator.</b> On S669 with homologues removed, pooled Pearson r is
{v('s669','A_tsu_only','common25'):.3f} / {v('s669','B_fp_only','common25'):.3f} /
{v('s669','D_finetuned','common25'):.3f} for the three regimes, against per-protein medians of
{v('s669','A_tsu_only','common25','per_prot_median_r'):.2f}–{v('s669','B_fp_only','common25','per_prot_median_r'):.2f}.
On Ssym the apparent FireProt advantage vanishes once shared folds are removed
({v('ssym','A_tsu_only','common25'):.3f} ≈ {v('ssym','B_fp_only','common25'):.3f} ≈
{v('ssym','D_finetuned','common25'):.3f} on the common-clean subset). The gap between
ranking mutations <i>within</i> a protein and calibrating <i>across</i> proteins is the
central quantitative fact these benchmarks expose.
</div>

<h2>1. Motivation</h2>
<p>Held-out splits inside a single corpus measure generalisation to unseen proteins of the
same provenance. They cannot measure what happens when the assay, the curation lineage and
the protein universe all change at once. S669 and Ssym are the two most widely used blind
stability benchmarks, and both are external to this project's training corpora — which makes
them the right instrument, provided sequence-identity leakage is controlled rather than
assumed away.</p>

<p>Two questions follow. Which training corpus transfers better to an external benchmark, a
large set of designed mini-domains or a smaller set of natural proteins? And does the
apparent answer survive removing benchmark proteins that are homologous to the training
set?</p>

<h2>2. Methods</h2>
<p><b>Benchmarks.</b> S669 ({n('s669','A_tsu_only','full')} variants over
{int(v('s669','A_tsu_only','full','n_prot'))} proteins) is diverse and deliberately dissimilar
to common training sets. Ssym ({n('ssym','A_tsu_only','full')} variants over
{int(v('ssym','A_tsu_only','full','n_prot'))} proteins) is narrow and dominated by a handful
of well-studied folds. Both are capped at 500 residues and every variant's position was
validated against the provided sequence. Ssym is direct mutations only; the reverse direction
is obtained analytically from the antisymmetry-augmented model.</p>

<p><b>Regimes.</b> A — trained on Tsuboyama alone (12,359 mutations / 412 proteins).
B — trained on FireProt alone (3,205 / 138). D — pretrained on Tsuboyama, warm-start
continued on FireProt. Features are the concatenated wild-type and mutant pooled pair-track
embeddings; the model is a 5-seed MLP ensemble with antisymmetry augmentation on every
training set.</p>

<p><b>Leakage control.</b> The wild-type sequences of both training corpora and the benchmark
are pooled and clustered with MMseqs2 at 25 % and 30 % identity, 80 % coverage. A benchmark
protein is leaky with respect to a corpus if it shares a cluster with any protein in it.
Three views are reported: <i>full</i> (everything), <i>filtered</i> (drop proteins homologous
to that regime's own training corpus), and <i>common</i> (drop proteins homologous to
<i>any</i> corpus, giving one identical variant subset for all three regimes — the only fair
cross-regime comparison).</p>

<p><b>Sign convention.</b> The benchmarks use the opposite ΔΔG sign; predictions are
sign-flipped when the pooled Pearson is negative, and the flip is recorded.</p>

<h2>3. Results</h2>

<h3>3.1 S669 — the hard test</h3>
<table>
<caption>Pooled Pearson r, with the per-protein median r in parentheses. Regime A has zero
overlap with S669, so its filtered column equals its full column.
Common-25 holds {n('s669','B_fp_only','common25')} variants.</caption>
<tr><th>regime</th><th>full ({n('s669','A_tsu_only','full')})</th><th>filtered (25 %)</th><th>common (25 %)</th></tr>
{bench_rows('s669', ('full', 'filt25', 'common25'))}
</table>

<p>Two things separate here. Pooled r — which asks whether the model gets the <i>magnitude</i>
right across different proteins — sits between {v('s669','A_tsu_only','common25'):.2f} and
{v('s669','B_fp_only','common25'):.2f} on the clean subset. The per-protein median, which asks
whether it ranks mutations correctly <i>inside</i> one protein, sits near
{v('s669','B_fp_only','common25','per_prot_median_r'):.2f} for every regime. The model orders
mutations within a fold considerably better than it places different folds on a common
scale.</p>

<h3>3.2 Ssym — where the corpus advantage turns out to be leakage</h3>
<table>
<caption>Pooled Pearson r (per-protein median). Common-25 holds only
{n('ssym','B_fp_only','common25')} variants, so it is a weak but unbiased comparison.</caption>
<tr><th>regime</th><th>full ({n('ssym','A_tsu_only','full')})</th><th>filtered (25 %)</th><th>common (25 %)</th></tr>
{bench_rows('ssym', ('full', 'filt25', 'common25'))}
</table>

<p>On the full set the FireProt-trained regime looks clearly best
({v('ssym','B_fp_only','full'):.3f} against {v('ssym','A_tsu_only','full'):.3f}). Removing
proteins that share a cluster with FireProt collapses that: its per-protein median falls from
{v('ssym','B_fp_only','full','per_prot_median_r'):.2f} to
{v('ssym','B_fp_only','filt25','per_prot_median_r'):.2f}, and on the common-clean subset the
three regimes are indistinguishable. <b>The advantage was homology, not training
distribution.</b></p>

<figure><img src="{F1}"/>
<figcaption><b>Figure 1.</b> Pooled Pearson r, full versus each regime's own homology filter.
Regime A has no benchmark overlap, so its two bars are equal by construction; the drop in B
and D is the size of the leakage.</figcaption></figure>

<h3>3.3 Antisymmetry: the model is nearly but not exactly antisymmetric</h3>
<p>Ssym pairs every forward mutation with its measured reverse, so it tests an identity the
model should satisfy: ΔΔG(A→B) = −ΔΔG(B→A). Correlation between the direct prediction and the
negated reverse is high for all three regimes
({ab('A_tsu_only','antisym_r'):.3f} / {ab('B_fp_only','antisym_r'):.3f} /
{ab('D_finetuned','antisym_r'):.3f}), but the residual <i>bias</i> is not negligible and it
differs by regime: {ab('A_tsu_only','bias_mean'):+.2f} kcal/mol for A,
{ab('B_fp_only','bias_mean'):+.2f} for B, {ab('D_finetuned','bias_mean'):+.2f} for D. Only
FireProt-only training is near-unbiased; training on designed mini-domains alone pushes
predictions toward destabilisation, and fine-tuning over-corrects past zero.</p>

<h3>3.4 An estimator defect, and what correcting it changed</h3>
<div class="caveat">
The benchmark model was originally fit with early stopping disabled and a fixed 250-iteration
budget. Because that budget counts <i>epochs</i>, the regime with the most training data took
several times as many gradient updates and over-trained hardest — biasing precisely the
cross-regime comparison the experiment exists to make. All numbers above use the project's
default estimator; the originals are retained for comparison.
</div>

<table>
<caption>Pooled Pearson r before and after the correction. The gain is systematically largest
for the regime with the most data, which is the signature the epoch-count argument predicts.</caption>
<tr><th>benchmark · subset</th><th>A Tsuboyama</th><th>B FireProt</th><th>D fine-tuned</th></tr>
<tr><td>S669 · full</td>
<td>{v('s669','A_tsu_only','full',table=pre):.3f} → <b>{v('s669','A_tsu_only','full'):.3f}</b></td>
<td>{v('s669','B_fp_only','full',table=pre):.3f} → {v('s669','B_fp_only','full'):.3f}</td>
<td>{v('s669','D_finetuned','full',table=pre):.3f} → {v('s669','D_finetuned','full'):.3f}</td></tr>
<tr><td>S669 · common-25</td>
<td>{v('s669','A_tsu_only','common25',table=pre):.3f} → <b>{v('s669','A_tsu_only','common25'):.3f}</b></td>
<td>{v('s669','B_fp_only','common25',table=pre):.3f} → {v('s669','B_fp_only','common25'):.3f}</td>
<td>{v('s669','D_finetuned','common25',table=pre):.3f} → {v('s669','D_finetuned','common25'):.3f}</td></tr>
<tr><td>Ssym · full</td>
<td>{v('ssym','A_tsu_only','full',table=pre):.3f} → {v('ssym','A_tsu_only','full'):.3f}</td>
<td>{v('ssym','B_fp_only','full',table=pre):.3f} → {v('ssym','B_fp_only','full'):.3f}</td>
<td>{v('ssym','D_finetuned','full',table=pre):.3f} → {v('ssym','D_finetuned','full'):.3f}</td></tr>
</table>

<p>The correction does not change which regime wins, but it halves the size of the corpus
effect on S669: the common-25 gap between the FireProt- and Tsuboyama-trained regimes falls
from {v('s669','B_fp_only','common25',table=pre) - v('s669','A_tsu_only','common25',table=pre):.3f}
to {v('s669','B_fp_only','common25') - v('s669','A_tsu_only','common25'):.3f}. It also
<b>reverses one conclusion</b>: on the defective numbers the fine-tuned regime had the best
S669 per-protein median, and on the corrected ones it is last
({v('s669','D_finetuned','common25','per_prot_median_r'):.2f} against
{v('s669','A_tsu_only','common25','per_prot_median_r'):.2f} and
{v('s669','B_fp_only','common25','per_prot_median_r'):.2f}). Fine-tuning does not earn its
keep here — which agrees with the separate within-FireProt result, where it also washed
out.</p>

<figure><img src="{F2}"/>
<figcaption><b>Figure 2.</b> (a) The correction, per benchmark, subset and regime; the gain
tracks training-set size rather than being uniform. (b) The residual antisymmetry bias on
Ssym's forward/reverse pairs, with its standard deviation.</figcaption></figure>

<h2>4. Interpretation</h2>
<p>Three claims survive both the homology filter and the estimator correction.</p>
<p><b>Leakage is large enough to invert a conclusion, and it is measurable.</b> Ssym's apparent
ranking of training corpora disappears entirely once shared folds are removed. Any benchmark
comparison that does not report an identity-controlled subset should be treated as
uninformative about generalisation.</p>
<p><b>The two benchmarks are not interchangeable.</b> Ssym is narrow and easy; every regime
scores between {min(v('ssym', r, 'full') for r in REGS):.2f} and
{max(v('ssym', r, 'full') for r in REGS):.2f} on it. S669 is diverse and hard, and it is the
one that discriminates.</p>
<p><b>Within-protein ranking and cross-protein calibration are different capabilities, and
this representation has much more of the first.</b> On clean S669 the per-protein median is
roughly {v('s669','B_fp_only','common25','per_prot_median_r'):.2f} while pooled r is roughly
{v('s669','A_tsu_only','common25'):.2f}. The defensible claim for this predictor is that it
ranks mutations within a fold; the cross-protein scale is where the error lives.</p>

<h2>5. Limitations</h2>
<ul>
<li>The common-clean Ssym subset holds only {n('ssym','B_fp_only','common25')} variants over a
handful of proteins. It is unbiased but weak, and the near-equality of the three regimes there
should not be read as a precise equivalence.</li>
<li>Both benchmarks are capped at 500 residues, so the absolute values are not directly
comparable to published figures computed on the complete sets, and no claim of
state-of-the-art performance is made.</li>
<li>Homology is controlled by sequence identity. Two proteins below 25 % identity can still
share a fold, so the filtered subsets bound sequence-level leakage, not structural
similarity.</li>
<li>Ssym's reverse direction is obtained from the model's own antisymmetry rather than from an
independent prediction, so §3.3 measures the internal consistency of the representation, not
agreement with an external reverse-mutation calculation.</li>
</ul>

<h2>6. Conclusion</h2>
<p>Asked how a frozen-trunk embedding ΔΔG predictor behaves on the field's standard blind
benchmarks, the answer depends almost entirely on whether homology is controlled and whether
the estimator is sound. With both handled, the picture is consistent: strong within-protein
ranking, weak cross-protein calibration, a narrow benchmark that cannot discriminate between
training regimes, and a diverse one on which the training-corpus effect is real but half the
size it first appeared.</p>

</body></html>"""

out_html = R / "_report.html"
out_html.write_text(HTML, encoding="utf-8")
subprocess.run(["wkhtmltopdf", "--enable-local-file-access", "--quiet",
                str(out_html), str(R / "report.pdf")], check=True)
out_html.unlink()
print(f"wrote {R / 'report.pdf'}")
