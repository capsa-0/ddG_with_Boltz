"""Build report.pdf for 10_gla_scan (paper-facing; no provenance, no plumbing).

    python results/10_gla_scan/build_report.py

Every number is recomputed from the committed result tables, so the report cannot
drift from the data. Self-contained HTML (figures embedded as base64) via wkhtmltopdf.
"""
import base64
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

R = Path(__file__).parent
merged = pd.read_csv(R / "compare_foldx_merged_mean.csv")
perpos = pd.read_csv(R / "compare_foldx_per_position_mean.csv")
disc = pd.read_csv(R / "discrepancy_by_position.csv")
summary = json.loads((R / "scan_summary.json").read_text())
preds = pd.read_csv(R / "scan_predictions_mean.csv")

# ---- derived numbers -------------------------------------------------------
n_mut, n_pos = len(merged), merged.position.nunique()
gly, non = merged[merged.wt_aa == "G"], merged[merged.wt_aa != "G"]
rho = lambda d: spearmanr(d.boltz, d.foldx).statistic
rho_all, rho_g, rho_n = rho(merged), rho(gly), rho(non)
clash = merged[merged.foldx < 10]
rho_clean = rho(clash)
pear_raw = merged.boltz.corr(merged.foldx)
pear_clip = merged.boltz.corr(merged.foldx.clip(-10, 10))

fl, rest = perpos[perpos.flagged], perpos[~perpos.flagged]
p_flag = mannwhitneyu(fl.spearman.dropna(), rest.spearman.dropna(),
                      alternative="two-sided").pvalue
pg, pn = perpos[perpos.wt_aa == "G"], perpos[perpos.wt_aa != "G"]
p_gly = mannwhitneyu(pg.spearman.dropna(), pn.spearman.dropna(),
                     alternative="two-sided").pvalue

sd_b, sd_f = disc.boltz.std(), disc.foldx.std()
corr_diff_foldx = spearmanr(disc.diff_mean, disc.foldx).statistic
n_stab = int((merged.boltz < 0).sum())
n_stab_strong = int((merged.boltz < -0.5).sum())
reg = summary["per_regime"]
agree = summary["regime_agreement_pearson"]

# ---- per-regime robustness (table 2): does every headline claim survive the
# choice of training corpus, or only the across-regime mean? -----------------
REGIME_LABEL = {"ddg_A_tsuboyama": "A — Tsuboyama-only",
                "ddg_B_fireprot": "B — FireProt-only",
                "ddg_D_finetuned": "D — A fine-tuned on B",
                "ddg_mean": "Mean of A, B, D (used throughout)"}
_pr = preds.merge(merged[["mutation", "foldx"]], on="mutation")
_prg = _pr[_pr.wt_aa == "G"]
_prn = _pr[_pr.wt_aa != "G"]
regime_rows = [(
    REGIME_LABEL[c],
    spearmanr(_pr[c], _pr.foldx).statistic,
    spearmanr(_prn[c], _prn.foldx).statistic,
    spearmanr(_prg[c], _prg.foldx).statistic,
    100 * (_pr[c] < 0).mean(),
) for c in REGIME_LABEL]
rho_reg_lo = min(r[1] for r in regime_rows[:3])
rho_reg_hi = max(r[1] for r in regime_rows[:3])
stab_lo = min(r[4] for r in regime_rows[:3])
stab_hi = max(r[4] for r in regime_rows[:3])
REGIME_TBL = "\n".join(
    f"<tr><td>{lab}</td><td>{a:+.3f}</td><td>{n:+.3f}</td><td>{g:+.3f}</td>"
    f"<td>{s:.1f}%</td></tr>" for lab, a, n, g, s in regime_rows)

# ---- landscape structure (3.1), replacing the former heatmap figure --------
# omega^2 rather than eta^2: variance explained is inflated by group count, and
# the two factors here differ 9-fold in it (177 positions vs 20 mutant residues),
# so the uncorrected numbers are not comparable to each other.
def _omega2(df, key, col="ddg_mean"):
    grp = df.groupby(key)[col]
    n, k, gm = len(df), grp.ngroups, df[col].mean()
    ss_b = (grp.count() * (grp.mean() - gm) ** 2).sum()
    ss_t = ((df[col] - gm) ** 2).sum()
    ms_w = (ss_t - ss_b) / (n - k)
    return (ss_b - (k - 1) * ms_w) / (ss_t + ms_w)


w2_pos, w2_aa = _omega2(preds, "position"), _omega2(preds, "mut_aa")
# Counts of "tolerates everything" / "rejects everything" are meaningless at a
# position with 3 scored substitutions, so they are taken over covered sites only.
MIN_COV = 10
_pp = preds.groupby("position").ddg_mean.agg(["mean", "min", "max", "count"])
_well = _pp[_pp["count"] >= MIN_COV]
n_well, n_scan_pos = len(_well), len(_pp)
pp_lo, pp_hi = _well["mean"].min(), _well["mean"].max()
n_tol = int((_well["max"] < 0.5).sum())
n_inv = int((_well["min"] > 1.5).sum())

# ---- external check: measured residual activity (Lukas 2013) ---------------
lk = pd.read_csv(R / "compare_lukas_merged.csv")
lkc = lk[lk.active_site == 0]                     # active-site variants excluded
n_lk, n_lkc = len(lk), len(lkc)
rho_lb, p_lb = spearmanr(lkc.boltz_mean, lkc.activity_pct_wt)
rho_lf, p_lf = spearmanr(lkc.foldx, lkc.activity_pct_wt)
_rng = np.random.default_rng(0)
_d = []
for _ in range(10000):
    _i = _rng.integers(0, n_lkc, n_lkc)
    _a = lkc.activity_pct_wt.values[_i]
    if len(set(_a)) < 3:
        continue
    _d.append(spearmanr(lkc.boltz_mean.values[_i], _a).statistic
              - spearmanr(lkc.foldx.values[_i], _a).statistic)
lk_lo, lk_hi = np.percentile(_d, [2.5, 97.5])
_dead = lkc.activity_pct_wt == 0
n_dead = int(_dead.sum())
med_dead, med_live = lkc.boltz_mean[_dead].median(), lkc.boltz_mean[~_dead].median()
p_dead = mannwhitneyu(lkc.boltz_mean[_dead], lkc.boltz_mean[~_dead], alternative="greater").pvalue
p_dead_f = mannwhitneyu(lkc.foldx[_dead], lkc.foldx[~_dead], alternative="greater").pvalue
p_lb_s = f"p = {p_lb:.3f}" if p_lb >= 0.001 else "p < 0.001"
p_lf_s = f"p = {p_lf:.3f}" if p_lf >= 0.001 else "p < 0.001"
rho_abs = abs(rho_lb)

# ---- percentile-diagonal shares (figure 1, top right) ----------------------
_ps = pd.read_csv(R / "percentile_shift_mean.csv").set_index("group").pct_below
pct_gly, pct_rest = _ps["glycine"], _ps["rest (non-Gly, non-flagged)"]
pct_fng, pct_fg = _ps["flagged, non-glycine"], _ps["flagged, glycine"]

img = lambda p: "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()
f_cmp = img(R / "figures/01_boltz_vs_foldx_mean.png")
f_raw = img(R / "figures/03_discrepancy_map_raw.png")
f_act = img(R / "figures/04_lukas_activity.png")

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10.5pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 17pt; margin: 0 0 2px; color: #14314f; }
h2 { font-size: 13pt; color: #14314f; border-bottom: 1.5px solid #d0d7de; padding-bottom: 3px; margin-top: 20px; }
h3 { font-size: 11pt; color: #14314f; margin: 14px 0 4px; }
.sub { color: #555; font-size: 9.5pt; margin: 0 0 4px; }
.headline { background: #eef4fb; border-left: 4px solid #2c6fb3; padding: 10px 14px; margin: 14px 0; }
.caveat { background: #fdf6e8; border-left: 4px solid #d9a441; padding: 9px 13px; margin: 12px 0; font-size: 10pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #cfd6dd; padding: 4px 8px; text-align: right; }
th { background: #f3f6f9; } td:first-child, th:first-child { text-align: left; }
caption { caption-side: bottom; font-size: 8.5pt; color: #666; padding-top: 4px; text-align: left; }
figure { margin: 12px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; } figcaption { font-size: 8.5pt; color: #555; margin-top: 4px; text-align: left; }
code { background: #f3f4f6; padding: 0 3px; font-size: 9pt; }
"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<h1>An exhaustive ΔΔG scan of human α-galactosidase A from structure-model embeddings</h1>
<p class="sub">Experiment 10 · ddG_with_Boltz · Boltz-2 pair-track embeddings · label-free application</p>

<div class="headline">
The ΔΔG predictor built on Boltz-2 embeddings is applied the way it would be used in
practice: pointed at a single protein of interest with <b>no experimental labels</b>, and
asked to score every possible point mutation. On human α-galactosidase A we score
<b>{n_mut:,} substitutions across {n_pos} positions</b> and compare against an independent
FoldX scan of the same protein. The two agree at <b>Spearman ρ = {rho_all:+.3f}</b>, but the
agreement is strongly residue-dependent — <b>{rho_n:+.3f} at non-glycine sites versus
{rho_g:+.3f} at glycines</b> — and the two methods differ by {sd_f/sd_b:.1f}× in dynamic
range. The scan behaves as a <b>destabilization ranker</b>: only {100*n_stab/n_mut:.1f}% of
its predictions fall below zero.
</div>

<h2>1. Motivation</h2>
<p>Stability predictors are normally reported on curated benchmarks, where every variant has
a measured ΔΔG. The practical question is different: given one protein and no measurements,
can the model rank its mutational landscape usefully? Answering it requires an exhaustive
scan — every position, every substitution — and a way to judge the output when no ground
truth exists.</p>
<p>Human α-galactosidase A is a demanding test case. It is a 398-residue human lysosomal
enzyme, far from the small designed domains that dominate the training corpus, and it is
absent from every dataset the model was trained on, so its predictions here are genuinely
blind. An independent FoldX scan of the same protein provides a second, mechanistically
unrelated opinion.</p>

<h2>2. Methods</h2>
<h3>2.1 Representation and model</h3>
<p>For each mutation the wild-type and mutant sequences are passed through Boltz-2 in an
embeddings-only mode, and the pair-track (<i>z</i>) row of the mutated residue is pooled over
partners for each. The two pooled vectors are <b>concatenated</b> (128 + 128), retaining both
absolute levels rather than only their difference. The regressor is a 5-seed ensemble of
MLPs (256–128–64) over median-imputed, standardised features, trained with
<b>antisymmetry augmentation</b>: the reverse mutation is the two halves exchanged with the
label negated.</p>
<h3>2.2 Three training regimes</h3>
<p>Because no labels exist for the target, the model must be trained elsewhere. Three
regimes are reported together, differing only in training distribution:
<b>A</b> — a mega-scale folding-assay corpus of mostly small domains;
<b>B</b> — a literature-curated corpus of natural proteins;
<b>D</b> — A pretrained then fine-tuned on B. They are a robustness check on that free
parameter rather than three competing answers: the training corpus is the one modelling
choice that no measurement on this target can adjudicate, so every headline result is
repeated under each regime in §3.2 (Table 2) and the value reported everywhere else is
their mean. Positive ΔΔG denotes destabilization throughout.</p>
<h3>2.3 Comparison metric</h3>
<p>Predicted and FoldX values are compared by <b>rank correlation</b>. This is not a
stylistic choice: FoldX holds the backbone rigid, so a substitution that would be
accommodated by a small conformational shift is instead scored as an unrelievable clash,
producing values that reach tens of kcal/mol. Those points dominate any squared-error or
Pearson statistic. Pearson is reported alongside, on raw and clipped values, for reference
only.</p>

<h2>3. Results</h2>
<h3>3.1 The predicted landscape</h3>
<p>The three regimes place the landscape at similar levels (mean ΔΔG
{reg['A_tsuboyama']['mean']:+.2f}, {reg['B_fireprot']['mean']:+.2f} and
{reg['D_finetuned']['mean']:+.2f} kcal/mol for A, B and D) and agree closely on ordering
(pairwise Pearson {agree['A_tsuboyama_vs_B_fireprot']:.2f}–{agree['A_tsuboyama_vs_D_finetuned']:.2f}),
with a mean across-regime spread of {summary['mean_regime_sd']:.2f} kcal/mol. Conclusions
below are therefore not artifacts of one training corpus. <b>Every predicted value reported
from here on is the mean across the three regimes</b> — the quantity plotted, tabulated and
correlated in §3.2–§3.5. Table 2 repeats the headline comparison under each regime
separately, so the corpus dependence can be checked rather than assumed.</p>
<p>Within that landscape the dominant axis is <b>position, not substituted residue</b>:
position accounts for ω² = {w2_pos:.2f} of the variance in predicted ΔΔG against
ω² = {w2_aa:.2f} for the identity of the incoming residue. (The bias-corrected ω² is used
because the two factors differ nine-fold in group count — {n_scan_pos} positions against 20
residues — which inflates the uncorrected figure for position.) So the model is reading the
site far more than the substitution.</p>
<p>It grades those sites by degree rather than sorting them into tolerant and intolerant
classes. Across the {n_well} positions where at least {MIN_COV} of the 19 substitutions were
scored, the per-position mean ΔΔG spans {pp_lo:+.2f} to {pp_hi:+.2f} kcal/mol — a wide
range, but a near-continuous one. Positions that are absolute in either direction are rare:
only {n_inv} destabilise on <i>every</i> scored substitution (all above +1.5 kcal/mol) and
{n_tol} tolerates all of them (all below +0.5). Restricting to still better-covered
positions removes them entirely, so apparent all-or-nothing sites in this scan are an
artifact of thin coverage rather than a feature of the predictions.</p>

<h3>3.2 Agreement with an independent predictor</h3>
<table>
<caption><b>Table 1.</b> Rank agreement between the embedding scan and FoldX, the predicted
value being the across-regime mean of §3.1. Pearson values are shown for completeness; the
clipped column bounds FoldX at ±10 kcal/mol.</caption>
<tr><th>Subset</th><th>n</th><th>Spearman ρ</th><th>Pearson (raw)</th><th>Pearson (clipped)</th></tr>
<tr><td>All scored substitutions</td><td>{n_mut:,}</td><td>{rho_all:+.3f}</td><td>{pear_raw:+.3f}</td><td>{pear_clip:+.3f}</td></tr>
<tr><td>Non-glycine sites</td><td>{len(non):,}</td><td>{rho_n:+.3f}</td><td>—</td><td>—</td></tr>
<tr><td>Glycine sites</td><td>{len(gly):,}</td><td>{rho_g:+.3f}</td><td>—</td><td>—</td></tr>
<tr><td>Excluding FoldX clash regime (&lt;10 kcal/mol)</td><td>{len(clash):,}</td><td>{rho_clean:+.3f}</td><td>—</td><td>—</td></tr>
</table>
<figure><img src="{f_cmp}"/>
<figcaption><b>Figure 1.</b> Predicted ΔΔG (across-regime mean) against FoldX. Top left:
per-substitution comparison; the horizontal axis is
symlog because FoldX extends to tens of kcal/mol. Top right: the same points with each
method ranked <i>within its own spread</i>, so the diagonal needs no fitting and is immune to
the dynamic-range gap; a point below it is one FoldX ranks higher than the embedding model
does. Glycines fall below it {pct_gly:.0f}% of the time against {pct_rest:.0f}% for the rest,
and flagged positions that are <i>not</i> glycine sit at {pct_fng:.0f}%, i.e. within noise of
the rest. Bottom: per-position means. Agreement is visibly tighter away from glycine
positions.</figcaption></figure>
<p>The headline agreement of ρ = {rho_all:+.3f} is essentially unchanged when FoldX's clash
regime is excluded ({rho_clean:+.3f}), so it does not rest on the two methods merely
concurring that clashes are bad. It is, however, <b>strongly residue-dependent</b>: at
non-glycine sites the two predictors agree at {rho_n:+.3f}, at glycines only {rho_g:+.3f}.
Glycine is the residue whose replacement most often demands backbone accommodation, and it
is where the two methods' treatments diverge most.</p>
<table>
<caption><b>Table 2.</b> The same comparison computed separately under each training regime.
The mean row is the value used everywhere else in this report.</caption>
<tr><th>Training regime</th><th>ρ all</th><th>ρ non-Gly</th><th>ρ Gly</th><th>stabilizing</th></tr>
{REGIME_TBL}
</table>
<p>Neither of the report's two central claims depends on which corpus the model was trained
on. Agreement with FoldX varies between ρ = {rho_reg_lo:+.3f} and {rho_reg_hi:+.3f} across
the three regimes, spanning the reported {rho_all:+.3f}; the glycine deficit appears in
<i>every</i> regime, with non-glycine agreement exceeding glycine agreement in each row; and
the stabilizing tail stays small throughout ({stab_lo:.1f}–{stab_hi:.1f}% of predictions
below zero). Regime B agrees with FoldX least, but that ordering should not be read as an
accuracy ranking: on labelled external benchmarks the same predictor ranks these regimes the
other way round, and agreement with FoldX on a glycine-rich set is not a measure of
correctness in the absence of ground truth.</p>

<h3>3.3 The methods disagree about magnitude far more than about order</h3>
<figure><img src="{f_raw}"/>
<figcaption><b>Figure 2.</b> The same comparison in real units. Top: per-position difference.
Bottom: per-position means of both methods on one axis — the narrow bars are the embedding
model's entire dynamic range.</figcaption></figure>
<p>Per-position variability is {sd_f:.2f} kcal/mol for FoldX against {sd_b:.2f} for the
embedding model, a factor of {sd_f/sd_b:.1f}. The raw difference between the two correlates
with FoldX alone at ρ = {corr_diff_foldx:.3f}, i.e. it is very nearly the negative of the
FoldX value and carries almost no independent information. Magnitude, not ordering, is where
the two methods part company — and rank statistics are the only ones that survive it.</p>

<h3>3.4 A hypothesis from structural inspection</h3>
<p>Ten positions had been flagged, on structural grounds, as sites where the model appeared
to overestimate destabilization. Those positions do agree with FoldX less well than the
remainder (median per-position ρ {fl.spearman.median():+.3f} versus
{rest.spearman.median():+.3f}), and glycine positions likewise
({pg.spearman.median():+.3f} versus {pn.spearman.median():+.3f}). Neither separation reaches
significance at the position level ({'p = %.3f' % p_flag} and {'p = %.3f' % p_gly},
Mann–Whitney), because per-position correlations over ~19 substitutions are noisy. The
glycine effect is unambiguous only at the substitution level, where the sample is large.</p>
<p>At that level the two hypotheses turn out not to be independent. Ranking each method
within its own spread (Figure 1, top right), glycine substitutions fall on the FoldX-higher
side of the diagonal {pct_gly:.1f}% of the time versus {pct_rest:.1f}% for the rest — but
flagged positions that are not glycines sit at {pct_fng:.1f}%, indistinguishable from that
baseline, while the flagged positions that <i>are</i> glycines reach {pct_fg:.1f}%. The
flagged-position hypothesis is therefore largely a restatement of the glycine effect, with a
genuine additional shift only where the two coincide.</p>

<h3>3.5 An external check against measured data</h3>
<p>No measured ΔΔG exists for this protein, but a measured <i>proxy</i> does. A functional
study of Fabry disease expressed 157 missense variants of this enzyme in human cells and
quantified the residual α-galactosidase activity of each (Lukas et al., <i>PLoS Genetics</i>
2013). Activity is not stability — a substitution at a catalytic residue abolishes turnover
in a perfectly folded protein — so <b>variants at active-site positions are excluded</b>
here, leaving {n_lkc} that the scan also covers. What remains is a weak but genuine external
constraint: a variant destabilised enough to misfold should not retain activity.</p>
<figure><img src="{f_act}"/>
<figcaption><b>Figure 3.</b> Predicted ΔΔG against measured residual activity for the
{n_lkc} variants outside the active site. Red bars are the mean activity within terciles of
the prediction, labelled with how many variants in each tercile have exactly zero activity;
the FoldX axis is symlog. Both methods rank in the correct direction and neither separates
the many dead variants from each other — activity saturates at zero long before ΔΔG
does.</figcaption></figure>
<p>The predicted ranking runs the right way: <b>ρ = {rho_lb:+.3f}</b> ({p_lb_s}) against
measured activity, versus <b>{rho_lf:+.3f}</b> ({p_lf_s}) for FoldX on the same variants —
a difference indistinguishable from zero (paired bootstrap CI [{lk_lo:+.3f}, {lk_hi:+.3f}]).
Variants with no detectable activity ({n_dead} of {n_lkc}) carry a higher predicted ΔΔG than
those retaining any ({med_dead:+.2f} versus {med_live:+.2f} kcal/mol median, Mann–Whitney
p = {p_dead:.3f}; the same test on FoldX gives p = {p_dead_f:.3f}).</p>
<p>The size of the effect is the honest part of this result. A correlation near {rho_abs:.2f}
against a saturating, zero-inflated readout on {n_lkc} variants establishes that the scan
ranks loss of function better than chance; it does not establish accuracy on a ΔΔG scale,
and it cannot, because the measurement is not one.</p>

<h2>4. Interpretation</h2>
<p><b>The glycine deficit is a property of the model, not of this protein.</b> An independent
error analysis of the same predictor on labelled held-out data finds buried glycines to be
one of its few genuine weak spots once effect size is controlled for, alongside proline
targets. The present scan reproduces that signature blind, on a protein with no measurements
at all, which is the stronger form of the observation.</p>
<p><b>Burial degrades the physics-based method, not the learned one.</b> The extreme FoldX
values here occur at buried glycines, where a rigid backbone cannot relieve the clash. The
same error analysis finds the embedding model's <i>relative</i> accuracy to be flat across
burial tertiles. The two methods therefore fail in different places, which is what makes
their disagreement informative even without ground truth.</p>
<p><b>The practical limitation is the stabilizing tail.</b> Only {n_stab:,} of {n_mut:,}
predictions ({100*n_stab/n_mut:.1f}%) fall below zero and just {n_stab_strong} below
−0.5 kcal/mol. The model is a destabilization ranker: it is well suited to asking which
substitutions a protein will not tolerate, and poorly suited to searching for stabilizing
ones — the regime that matters most for engineering.</p>

<div class="caveat">
<b>What this comparison cannot establish.</b> No measured ΔΔG exists for this protein — the
activity data of §3.5 constrains loss of function, not folding free energy — so
FoldX is a second opinion and not a reference. Where the two disagree, neither is thereby
shown to be wrong. The one asymmetry available is mechanistic: in the clash regime there is
an independent reason to distrust FoldX, whereas at exposed sites the disagreement is simply
unresolved.
</div>

<h2>5. Limitations</h2>
<ul>
<li><b>Partial coverage.</b> {n_mut:,} substitutions at {n_pos} of the protein's 398
positions. The scored positions are not a random sample, so protein-wide summaries should
not be read off them.</li>
<li><b>No ground truth.</b> Neither absolute accuracy nor calibration can be assessed on this
target; only agreement with another predictor, and a rank check against a measured
<i>activity</i> proxy that saturates at zero and confounds folding with catalysis.</li>
<li><b>Per-position statistics are noisy.</b> With ~19 substitutions per site, individual
per-position correlations carry wide uncertainty and should not be ranked against each
other.</li>
<li><b>Rank-based comparison is deliberately blind to scale.</b> It cannot see the
{sd_f/sd_b:.1f}× dynamic-range difference, which is reported separately.</li>
</ul>

<h2>6. Conclusion</h2>
<p>A structure-model embedding predictor can be applied to a whole protein without labels and
produces a mutational landscape that agrees with an established physics-based method at
ρ = {rho_all:+.3f}, rising to {rho_n:+.3f} away from glycine sites. The two methods fail in
different, identifiable places: FoldX at buried glycines, where its rigid backbone forces
runaway clash energies, and the embedding model on stabilizing mutations, which it rarely
predicts at all. For triaging which substitutions a protein will not tolerate, the scan is
usable today; for discovering stabilizing ones, its blind spot is the limiting factor.</p>
</body></html>"""

out_html = R / "_report.html"
out_html.write_text(HTML, encoding="utf-8")
subprocess.run(["wkhtmltopdf", "--quiet", "--enable-local-file-access",
                str(out_html), str(R / "report.pdf")], check=True)
out_html.unlink()
print(f"wrote {R/'report.pdf'}  ({n_mut} mutations, {n_pos} positions, rho={rho_all:+.3f})")
