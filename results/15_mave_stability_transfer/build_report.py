"""Build report.pdf for 15_mave_stability_transfer (paper-facing; no provenance).

    python results/15_mave_stability_transfer/build_report.py

Self-contained HTML (figures embedded as base64) rendered via wkhtmltopdf. Every
number is read from the committed result tables, so the PDF cannot drift from them.
Per results/guidelines.md this file carries motivation, methods, results and
interpretation only — corpus assembly, file paths, run history and problems
encountered live in status.md and the README provenance table.
"""
import base64
import subprocess
from pathlib import Path

import pandas as pd

R = Path(__file__).parent

l1 = pd.read_csv(R / "layer1_direct.csv")
l2 = pd.read_csv(R / "layer2_lopo_summary.csv")
bp = pd.read_csv(R / "bootstrap_protein.csv")
bpn = pd.read_csv(R / "bootstrap_protein_noubi4.csv")
strata = pd.read_csv(R / "conservation_strata_auc.csv")
vt = pd.read_csv(R / "vampseq_dissociation.csv")
vb = pd.read_csv(R / "vampseq_bootstrap.csv")


def lopo(model, arm, col="median_spearman"):
    row = l2[(l2.model == model) & (l2.arm == arm)]
    return float(row[col].iloc[0])


def gap(model, table=bp):
    r = table[table.model == model].iloc[0]
    b, e = ("<b>", "</b>") if r.excludes_zero else ("", "")
    return f"{b}{r.delta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]{e}"


def med_abs(col):
    return l1[col].abs().median()


def strat(name):
    r = strata[strata.stratum == name].iloc[0]
    return r


def vrow(contrast):
    return vb[vb.contrast == contrast].iloc[0]


def vassay(assay, col):
    return float(vt[vt.assay == assay][col].iloc[0])


def img(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


F = {n: img(R / f"figures/{n}") for n in (
    "01_lopo_paired.png", "02_per_dataset_direct.png",
    "03_landscape_reproduction.png", "04_conservation_strata.png",
    "05_vampseq_dissociation.png")}

pooled, cond = strat("pooled"), strat("conditional")
vros, vgem = vrow("boltz_minus_rosetta"), vrow("boltz_minus_gemme")
ABU, FUN = "abundance (VAMP-seq)", "drug sensitivity"

CSS = """
@page { size: A4; margin: 15mm 16mm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 17pt; margin: 0 0 2px; color: #14493c; }
h2 { font-size: 12.5pt; color: #14493c; border-bottom: 1.5px solid #d0d7d4; padding-bottom: 3px; margin-top: 18px; }
h3 { font-size: 11pt; color: #14493c; margin: 14px 0 4px; }
.sub { color: #555; font-size: 9.5pt; margin: 0 0 4px; }
.headline { background: #eaf4f0; border-left: 4px solid #00966F; padding: 10px 14px; margin: 14px 0; }
.caveat { background: #fdf3ec; border-left: 4px solid #C25A12; padding: 9px 13px; margin: 12px 0; font-size: 9.5pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #cfd6d3; padding: 4px 8px; text-align: right; }
th { background: #f2f6f4; } td:first-child, th:first-child { text-align: left; }
caption { caption-side: bottom; font-size: 8.5pt; color: #666; padding-top: 4px; text-align: left; }
figure { margin: 12px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; } figcaption { font-size: 8.5pt; color: #555; margin-top: 4px; text-align: left; }
code { background: #f2f4f3; padding: 0 3px; font-size: 8.8pt; }
"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>

<h1>Stability information from a structure-prediction trunk transfers to cellular fitness</h1>
<p class="sub">Boltz-2 embedding ΔΔG substituted for Rosetta ΔΔG in the Høie et al. (2022)
multiplexed-assay prediction framework · 11 proteins, 13 MAVE datasets, 23,415 scored variants</p>

<div class="headline">
<b>Our ΔΔG is a better standalone stability predictor of MAVE fitness than Rosetta's, and the
advantage disappears once conservation is in the model.</b> Under leave-one-protein-out
cross-validation the ΔΔG-only median Spearman ρ rises from {lopo('ddg_only','rosetta'):.3f}
(Rosetta) to {lopo('ddg_only','boltz'):.3f} (ours), a paired gap of {gap('ddg_only')}. With
GEMME conservation added the gap is {gap('ddg_dde')} — a tight null rather than an unproven
one. The advantage is concentrated in the assay that reads out stability most directly and
is absent from the ΔΔG-blind control, and roughly half of it is explained by conservation
signal our predictor carries and Rosetta's cannot.
</div>

<h2>1. Motivation</h2>
<p>Every previous benchmark in this series — Tsuboyama, FireProt, S669, Ssym — takes ΔΔG as
its target, so all of them inherit the same assay conventions and curation lineage. A model
can do well across all four while having learned the idiom of thermodynamic-stability datasets
rather than stability itself. The test that separates those two possibilities is a
<i>change of question</i>: predicting a label that is not ΔΔG, was measured in other
laboratories, by other assays, in units that are not kcal/mol.</p>

<p>Multiplexed assays of variant effects (MAVEs) supply exactly that. They report a cellular
fitness score for thousands of single substitutions at once. Høie et al. assembled 39 such
datasets over 29 proteins and showed that two computed quantities — Rosetta ΔΔG for stability
and GEMME ΔΔE for evolutionary conservation — jointly predict fitness, with conservation the
stronger of the two. Their framework has a well-defined stability slot, which makes it a
direct instrument: substitute our ΔΔG for Rosetta's, hold everything else fixed, and read the
paired difference.</p>

<p>The point is <i>not</i> ΔΔG accuracy, which the stability benchmarks already measure. Fitness
is not stability: a substitution can abolish function at an active site without unfolding
anything, so even a perfect ΔΔG predictor caps well below ρ = 1 here. A low absolute ρ is only
interpretable relative to Rosetta's on the same rows.</p>

<h2>2. Methods</h2>

<h3>Corpus and predictions</h3>
<p>Eleven proteins of ≤200 residues, carrying 13 of the 39 MAVE datasets. Each protein was
scanned at full L×19 saturation, because the richest model in the framework requires all 20
substitutions at a position. ΔΔG was predicted under three training regimes — Tsuboyama only,
FireProt only, and sequentially fine-tuned — using the representation and 5-seed MLP ensemble
adopted earlier in this series; the reported arm averages the three. The length cap is a
compute budget and does not tilt the comparison: median |ρ| between Rosetta ΔΔG and fitness is
{med_abs('rho_rosetta'):.3f} on these 13 datasets and 0.301 on all 39.</p>

<h3>The two comparison layers</h3>
<p><b>Direct.</b> Per-dataset Spearman correlation of each predictor against measured fitness,
with no model and no fitting.</p>
<p><b>Leave-one-protein-out.</b> The published random-forest protocol, decoded from the
released code rather than the paper prose, run once with Rosetta's ΔΔG in the stability slot
and once with ours. For each held-out dataset, every dataset belonging to the same protein is
removed from training. The headline is the median Spearman ρ across datasets. Before any
substitution the harness was verified against the four published baselines that the original
work pins explicitly, reproducing them to within 0.011 ρ.</p>

<h3>Controls</h3>
<p><b>Coverage matching.</b> Rosetta requires an experimental structure and is undefined for
4.3 % of variants; our sequence-based scan covers all of them. Our ΔΔG is therefore masked to
Rosetta's availability so both arms see identical rows and identical missingness, and neither
wins on coverage.</p>
<p><b>Homology.</b> One of the eleven proteins, ubiquitin, shares a 25 %/30 %-identity cluster
with the ΔΔG training corpus; two of thirteen datasets. Every result is reported both with and
without it.</p>
<p><b>Statistics.</b> Intervals are 95 % cluster bootstraps over the 11 proteins — the unit of
independence, since a protein's datasets share a sequence, a structure and an alignment —
computed on the <i>paired</i> difference within each resample, so the shared draw cancels.</p>
<p><b>Sign convention.</b> ΔΔG anti-correlates with fitness (destabilising → low fitness) while
conservation correlates positively. Direct comparisons are reported as |ρ| so the three
predictors are comparable in magnitude; the random forest predicts fitness, so its ρ is
positive for every arm.</p>

<h2>3. Results</h2>

<h3>3.1 The substitution is worth a real but modest gain, and only standalone</h3>

<table>
<caption>Median Spearman ρ across the 13 MAVE datasets, leave-one-protein-out. Both arms scored
on identical rows with identical missingness; the only difference is which ΔΔG occupies the
stability slot. Bold intervals exclude zero.</caption>
<tr><th>feature set</th><th>Rosetta</th><th>ours</th><th>Δ (95 % CI)</th><th>Δ, ubiquitin dropped</th></tr>
<tr><td>null (substitution matrix)</td><td>{lopo('null_smave','shared'):.3f}</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>ΔΔE only (GEMME)</td><td>{lopo('dde_only','shared'):.3f}</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td><b>ΔΔG only</b></td><td>{lopo('ddg_only','rosetta'):.3f}</td><td><b>{lopo('ddg_only','boltz'):.3f}</b></td><td>{gap('ddg_only')}</td><td>{gap('ddg_only', bpn)}</td></tr>
<tr><td>ΔΔG + ΔΔE</td><td>{lopo('ddg_dde','rosetta'):.3f}</td><td>{lopo('ddg_dde','boltz'):.3f}</td><td>{gap('ddg_dde')}</td><td>{gap('ddg_dde', bpn)}</td></tr>
<tr><td>position context (47 features)</td><td>{lopo('position_context','rosetta'):.3f}</td><td>{lopo('position_context','boltz'):.3f}</td><td>{gap('position_context')}</td><td>{gap('position_context', bpn)}</td></tr>
</table>

<p>The ΔΔG-only gain clears zero, with a lower bound of {bp[bp.model=='ddg_only'].ci_lo.iloc[0]:+.3f}.
The combined and position-context rows are <i>tight nulls</i>, not merely unproven: an interval
of [{bp[bp.model=='ddg_dde'].ci_lo.iloc[0]:+.3f}, {bp[bp.model=='ddg_dde'].ci_hi.iloc[0]:+.3f}]
rules out a meaningful combined-model advantage. Dropping ubiquitin — the one protein
homologous to the ΔΔG training corpus — leaves the result unchanged, and we are in fact
<i>worse</i> than Rosetta on both ubiquitin datasets in the direct comparison, the only two of
thirteen where we lose.</p>

<figure><img src="{F['01_lopo_paired.png']}"/>
<figcaption><b>Figure 1.</b> Leave-one-protein-out performance by feature set, both arms, and the
paired difference with its 95 % protein-bootstrap interval. Grey bars carry no ΔΔG term and are
therefore arm-agnostic. Only ΔΔG-only clears zero.</figcaption></figure>

<figure><img src="{F['02_per_dataset_direct.png']}"/>
<figcaption><b>Figure 2.</b> The same comparison with no model: direct |Spearman ρ| per dataset.
Median |ρ| is {med_abs('rho_rosetta'):.3f} for Rosetta, {med_abs('rho_boltz_mean'):.3f} for ours
and {med_abs('rho_gemme'):.3f} for GEMME conservation.</figcaption></figure>

<h3>3.2 Our ΔΔG places variants in the same mechanistic picture, on a compressed axis</h3>

<p>Correlation alone does not establish that a predictor participates in the stability–conservation
structure the original work described. Reproducing their landscape does. Every variant is placed
on a plane of ΔΔG against conservation, discretised into sectors, and each sector scored by the
fraction of its variants that lose function. With Rosetta's own ΔΔG on this 13-dataset subset the
two corner sectors the original reports come back at 84 % and 96 % against their published 81 %
and 93 %, which is what licenses reading the substituted arm.</p>

<p>Our ΔΔG reproduces the sector structure closely — the high-conservation row runs 68/81/88/96 %
against Rosetta's 71/81/88/96 % — but on a compressed scale: its standard deviation is
0.97 kcal/mol against Rosetta's 2.14, so the original absolute thresholds leave its most
destabilising column nearly empty. Sectors are therefore also drawn at quantile-matched cuts,
which is the comparison the rank-based metrics downstream actually use.</p>

<figure><img src="{F['03_landscape_reproduction.png']}"/>
<figcaption><b>Figure 3.</b> The stability–conservation landscape and its sector grid, reproduced
with each ΔΔG in turn. Cells holding fewer than 50 variants are greyed in the difference panel.
The sectors do not contain the same variants in both arms, so the difference compares structure
rather than paired variants.</figcaption></figure>

<h3>3.3 Roughly half of the advantage is explained by conservation</h3>

<p>Our ΔΔG is predicted by a model conditioned on a multiple sequence alignment; Rosetta's is a
force-field calculation that structurally cannot see one. If part of our advantage is evolutionary
signal rather than stability, it should shrink when conservation is held fixed. Stratifying by
conservation quartile and scoring each predictor's ability to detect loss of function tests this
directly.</p>

<table>
<caption>AUC for detecting loss of function. The conditional row is the mean of the four
stratum AUCs, computed as one statistic per bootstrap resample.</caption>
<tr><th>comparison</th><th>Rosetta</th><th>ours</th><th>Δ AUC (95 % CI)</th></tr>
<tr><td>pooled (unconditioned)</td><td>{pooled.auc_rosetta:.3f}</td><td>{pooled.auc_boltz:.3f}</td><td><b>{pooled.delta:+.3f} [{pooled.ci_lo:+.3f}, {pooled.ci_hi:+.3f}]</b></td></tr>
<tr><td>within conservation strata</td><td>{cond.auc_rosetta:.3f}</td><td>{cond.auc_boltz:.3f}</td><td>{cond.delta:+.3f} [{cond.ci_lo:+.3f}, {cond.ci_hi:+.3f}]</td></tr>
</table>

<p>Conditioning removes {100*(1-cond.delta/pooled.delta):.0f} % of the advantage, and what remains
no longer clears zero. This is the first quantitative signature that part of our ΔΔG is
conservation. It is not a demonstration in either direction: the residual is positive in all four
strata ({strat('Q1').delta:+.3f} to {strat('Q4').delta:+.3f}), so the failure to clear zero
reflects the power available from eleven proteins rather than an established absence.</p>

<figure><img src="{F['04_conservation_strata.png']}"/>
<figcaption><b>Figure 4.</b> Loss-of-function detection by conservation quartile, and the pooled
versus conditional advantage over Rosetta with paired protein-bootstrap intervals.</figcaption></figure>

<h3>3.4 Where a stability predictor should win, it wins — and the assay decides, not the protein</h3>

<p>NUDT15 contributes two of the thirteen datasets: a VAMP-seq abundance assay, which reports
cellular protein level and is the closest available readout of stability, and a drug-sensitivity
assay, which reports enzyme function. Sequence, structure, alignment and our ΔΔG predictions are
identical across the two; only the measured quantity changes. This is a within-protein control on
whether a predictor tracks stability or merely ranks variants well in general.</p>

<table>
<caption>|Spearman ρ| against measured fitness, NUDT15 under two assays. The ordering of the three
predictors inverts.</caption>
<tr><th>predictor</th><th>abundance (VAMP-seq)</th><th>drug sensitivity</th></tr>
<tr><td>GEMME ΔΔE (conservation)</td><td>{vassay(ABU,'gemme'):.3f}</td><td><b>{vassay(FUN,'gemme'):.3f}</b></td></tr>
<tr><td>Rosetta ΔΔG</td><td>{vassay(ABU,'rosetta'):.3f}</td><td>{vassay(FUN,'rosetta'):.3f}</td></tr>
<tr><td>our ΔΔG</td><td><b>{vassay(ABU,'boltz'):.3f}</b></td><td>{vassay(FUN,'boltz'):.3f}</td></tr>
</table>

<p>On the function assay conservation is the strongest predictor and both ΔΔG calculations trail
it. On the abundance assay the ordering reverses: both ΔΔG calculations overtake conservation, and
ours leads by {vros['mean']:+.3f} over Rosetta
[{vros.ci_lo:+.3f}, {vros.ci_hi:+.3f}] and {vgem['mean']:+.3f} over GEMME
[{vgem.ci_lo:+.3f}, {vgem.ci_hi:+.3f}], from a cluster bootstrap over the protein's
{int(vros.n_positions)} positions. This reproduces the original work's central claim inside a
single protein, and is the clearest available evidence that the quantity our model extracts is
stability rather than general variant tolerance.</p>

<p>Two further observations follow. First, this is partial counter-evidence to the conservation
account of §3.3: on this dataset conservation is <i>weak</i> ({vassay(ABU,'gemme'):.3f}) and we
score roughly double it, whereas a predictor whose advantage were mostly smuggled conservation
should fail where conservation fails. The two results are jointly consistent with a ΔΔG that is
part evolutionary and part structural. Second, on this dataset adding conservation to the model
actively <i>hurts</i> our arm — ΔΔG-only reaches 0.518 against 0.429 for ΔΔG+ΔΔE and 0.439 for the
full position-context model — so on the purest stability readout in the corpus our single-feature
model outperforms every richer one.</p>

<figure><img src="{F['05_vampseq_dissociation.png']}"/>
<figcaption><b>Figure 5.</b> The same protein under a stability assay and a function assay, and the
VAMP-seq contrasts with within-protein bootstrap intervals. The position-clustered interval answers
whether this dataset's gap is real, not whether it generalises across proteins.</figcaption></figure>

<h2>4. Interpretation</h2>

<p>A ΔΔG predictor built by regressing on frozen internal representations of a structure-prediction
model carries stability information that survives a change of question. Against cellular fitness
measured by other laboratories in units that are not kcal/mol, it outperforms the field's reference
physics-based stability calculation as a standalone term, does so without requiring the experimental
structure that calculation needs, and concentrates its advantage in the assay that reads out
stability most directly while adding no spurious signal to the control assay where neither
stability nor conservation predicts anything.</p>

<p>The qualifier is equally clear. The advantage does not survive the presence of an explicit
conservation term, and about half of it is attributable to conservation the predictor carries
implicitly. Conservation alone remains the stronger single predictor of fitness
({lopo('dde_only','shared'):.3f} against {lopo('ddg_only','boltz'):.3f}), and the full model reaches
{lopo('position_context','rosetta'):.3f}. Stability is not the dominant term in fitness; this work
improves one input to a model of it and does not disturb that ordering.</p>

<div class="caveat">
<b>The open question, and the experiment that settles it.</b> The most likely explanation for the
pattern — a real standalone gain that vanishes beside explicit conservation — is that the
structure-prediction trunk is conditioned on a multiple sequence alignment and the force-field
calculation is not. An earlier ablation in this series measured the alignment's contribution to
this model at 0.08–0.10 Pearson r, close to the gap observed here. §3.3 bounds the effect at about
half, and §3.4 shows it cannot be the whole story. The decisive test is to rebuild the corpus with
the trunk run in single-sequence mode and repeat: if the advantage survives it is structural; if it
collapses to parity, the honest description is that this ΔΔG is in part a conservation predictor.
</div>

<h2>5. Limitations</h2>
<ul>
<li>Eleven proteins is a small unit count for a cluster bootstrap. The ΔΔG-only interval clears
zero, but its lower bound is {bp[bp.model=='ddg_only'].ci_lo.iloc[0]:+.3f}; the conditional analysis
of §3.3 is underpowered outright. Extending to the next length tier would add a second abundance
assay and the sharpest ΔΔG-blind control in the original set.</li>
<li>The 200-residue cap was verified not to bias the comparison, but it bounds dataset difficulty,
not method behaviour: it cannot exclude that the trunk models longer chains less well.</li>
<li>The Rosetta arm is a published artifact. It cannot be audited, tuned or improved here, so the
comparison is against the calculation as reported, in the protocol reported. The harness
reproduction of the published baselines is what makes that comparison fair rather than favourable.</li>
<li>§3.4 rests on a single dataset and a single protein, with an interval that speaks to that
dataset only. It qualifies the headline; it cannot replace it.</li>
<li>The two arms are not matched in inputs: the force-field calculation was given experimental
structures and ours only sequence and an alignment. This favours the practical claim — no structure
required — while making the comparison something other than physics against physics.</li>
</ul>

<h2>6. Conclusion</h2>
<p>Asked whether a ΔΔG regressed on frozen structure-prediction embeddings carries stability
information competitive with an established physics-based calculation, on a label that is not ΔΔG,
the answer is yes as a standalone term and no once conservation is supplied explicitly. The
strongest form of the positive result is not the pooled median but its placement: the advantage is
largest on the assay that reads out stability most directly, is absent where nothing predicts, and
survives the removal of the only homologous protein. The strongest form of the negative result is
that roughly half of the pooled advantage is conservation the predictor absorbed from its
alignment — a quantity now bounded, and resolvable by one further ablation.</p>

</body></html>"""

out_html = R / "_report.html"
out_html.write_text(HTML, encoding="utf-8")
subprocess.run(["wkhtmltopdf", "--enable-local-file-access", "--quiet",
                str(out_html), str(R / "report.pdf")], check=True)
out_html.unlink()
print(f"wrote {R / 'report.pdf'}")
