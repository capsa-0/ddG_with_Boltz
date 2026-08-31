"""Compare the GLA scan predictions against measured residual enzyme activity.

Ground truth for ΔΔG does not exist for α-galactosidase A (see status.md). The
closest measured quantity is the residual activity of Fabry missense variants
expressed in HEK293H cells, from:

    Lukas J. et al. (2013) "Functional Characterisation of Alpha-Galactosidase A
    Mutations as a Basis for a New Classification System in Fabry Disease",
    PLoS Genetics 9(8):e1003632 — Supplementary Table S1.

Activity is NOT stability: a variant can lose activity by mutating a catalytic
residue while remaining perfectly folded. So this is an *ordinal proxy* and the
only defensible test is a rank correlation with the expected sign (higher
predicted ΔΔG -> lower residual activity), reported alongside the same test for
FoldX on the same mutations.

    python results/10_gla_scan/compare_lukas.py            # uses committed CSV
    python results/10_gla_scan/compare_lukas.py --fetch    # re-download + re-parse S1
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
S1_URL = ("https://journals.plos.org/plosgenetics/article/file"
          "?type=supplementary&id=10.1371/journal.pgen.1003632.s003")

# ---------------------------------------------------------------------------
# Active site of human α-Gal A, in UniProt P06280 numbering (which PDB 1R47 also
# uses for the mature chain, 32-429, so no offset is needed anywhere here).
#
# Two independent sources, unioned, both verifiable:
#   (a) UniProt P06280 feature table — ACT_SITE 170 "Nucleophile", ACT_SITE 231
#       "Proton donor", BINDING 203..207 "substrate".
#   (b) every residue with a heavy atom within LIGAND_CUTOFF of the galactose
#       bound in the catalytic pocket of PDB 1R47 (ligand GAL A1101 / B1103;
#       the NAG/MAN/FUC hetero-atoms are N-glycans and are excluded), unioned
#       over both monomers of the homodimer.
#
# Recompute (b) and check the constant with:  compare_lukas.py --pdb 1R47.pdb
# Variants at these positions can be catalytically dead while perfectly folded,
# which is exactly the confound that makes activity an imperfect stability proxy.
UNIPROT_SITES = {170, 231} | set(range(203, 208))
LIGAND_CUTOFF = 5.0
ACTIVE_SITE = {47, 92, 93, 134, 142, 143, 168, 170,
               203, 204, 205, 206, 207, 227, 231, 266, 267}


def active_site_from_pdb(pdb_path: Path, cutoff: float = LIGAND_CUTOFF) -> set[int]:
    """Residues within `cutoff` A of the catalytic-pocket galactose in 1R47."""
    prot: dict[int, list] = {}
    lig = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")) or line[76:78].strip() == "H":
            continue
        chain, resn = line[21], line[17:20].strip()
        if chain not in "AB":
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if line.startswith("ATOM"):
            prot.setdefault(int(line[22:26]), []).append(xyz)
        elif resn == "GAL":                    # catalytic ligand, not the glycans
            lig.append(xyz)
    if not lig:
        raise SystemExit(f"no GAL ligand in {pdb_path} — is this 1R47?")
    lig = np.asarray(lig)
    return {num for num, atoms in prot.items()
            if np.linalg.norm(np.asarray(atoms)[:, None, :] - lig[None, :, :],
                              axis=2).min() <= cutoff}

MUT_RE = re.compile(r"^p\.([A-Z])(\d{1,3})([A-Z])$")


def _doc_text(blob: bytes) -> str:
    """Extract the text of a Word-97 .doc (OLE + piece table). No external tools."""
    import io

    import olefile

    ole = olefile.OleFileIO(io.BytesIO(blob))
    wd = ole.openstream("WordDocument").read()
    table = "1Table" if (struct.unpack_from("<H", wd, 0x0A)[0] >> 9) & 1 else "0Table"
    tb = ole.openstream(table).read()
    fc_clx, lcb_clx = struct.unpack_from("<II", wd, 0x1A2)
    clx = tb[fc_clx:fc_clx + lcb_clx]
    i = 0
    while clx[i] == 1:                                   # skip the Prc blocks
        i += 3 + struct.unpack_from("<H", clx, i + 1)[0]
    plc = clx[i + 5:i + 5 + struct.unpack_from("<I", clx, i + 1)[0]]
    n = (len(plc) - 4) // 12
    cps = [struct.unpack_from("<I", plc, 4 * k)[0] for k in range(n + 1)]
    out = []
    for k in range(n):
        fc = struct.unpack_from("<I", plc, 4 * (n + 1) + 8 * k + 2)[0]
        compressed = bool(fc & 0x40000000)
        start = (fc & 0x3FFFFFFF) // 2 if compressed else fc
        ln = cps[k + 1] - cps[k]
        raw = wd[start:start + (ln if compressed else 2 * ln)]
        out.append(raw.decode("cp1252", "replace") if compressed
                   else raw.decode("utf-16le", "replace"))
    return "".join(out)


def fetch_lukas(dest: Path) -> None:
    """Download Table S1 and write the parsed missense rows to `dest`."""
    import urllib.request

    req = urllib.request.Request(S1_URL, headers={"User-Agent": "Mozilla/5.0"})
    text = _doc_text(urllib.request.urlopen(req, timeout=120).read())

    def num(cell: str) -> float | None:
        m = re.match(r"^([\d.]+)", cell.replace("<", "").strip())
        return float(m.group(1)) if m else None

    rows = []
    for block in text.split("\x07\x07"):                 # \x07 = cell, \x07\x07 = row
        cells = [c.strip().replace("\r", " ") for c in block.split("\x07")]
        if not cells or not MUT_RE.match(cells[0]):
            continue
        rows.append({"mutation": cells[0][2:],
                     "act_minus_DGJ": num(cells[2]) if len(cells) > 2 else None,
                     "act_plus_DGJ": num(cells[3]) if len(cells) > 3 else None,
                     "raw_minus_DGJ": cells[2] if len(cells) > 2 else ""})
    rows = [r for r in rows if r["act_minus_DGJ"] is not None]
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"fetched {len(rows)} missense variants -> {dest.name}")


def load(path: Path, key: str, val: str) -> dict[str, float]:
    return {r[key]: float(r[val]) for r in csv.DictReader(path.open()) if r[val]}


def spearman(x, y) -> tuple[float, float]:
    r = stats.spearmanr(x, y)
    return r.statistic, r.pvalue


def paired_boot(b, f, a, n=10000, seed=0) -> tuple[float, float, float, float]:
    """Bootstrap the difference rho(b, a) - rho(f, a) over the same resampled rows."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i])) < 3:
            continue
        diffs.append(stats.spearmanr(b[i], a[i]).statistic
                     - stats.spearmanr(f[i], a[i]).statistic)
    diffs = np.asarray(diffs)
    obs = stats.spearmanr(b, a).statistic - stats.spearmanr(f, a).statistic
    return (float(obs), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)), float(np.mean(diffs < 0)))


def boot_rho(x, y, n=10000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if len(set(y[i])) < 3:
            continue
        vals.append(stats.spearmanr(x[i], y[i]).statistic)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="re-download and re-parse Table S1")
    ap.add_argument("--pdb", type=Path, help="1R47.pdb — recompute the active-site shell")
    ap.add_argument("--scan", type=Path, default=HERE / "scan_predictions_mean.csv")
    # results/10's original scan carried one column per training regime; the results/16
    # transfer model (`diag`) emits a single prediction, so the column is a parameter.
    ap.add_argument("--ddg-col", default="ddg_mean",
                    help="prediction column in --scan (e.g. ddg_diag)")
    ap.add_argument("--foldx", type=Path, default=HERE / "ddg_varmed_by_mutation_foldx.csv")
    ap.add_argument("--activity", type=Path, default=HERE / "lukas2013_activity.csv")
    ap.add_argument("--out", type=Path, default=HERE / "compare_lukas_merged.csv")
    ap.add_argument("--figure", type=Path, default=HERE / "figures" / "04_lukas_activity.png")
    args = ap.parse_args()

    if args.fetch:
        fetch_lukas(args.activity)

    if args.pdb:
        shell = active_site_from_pdb(args.pdb)
        derived = shell | UNIPROT_SITES
        print(f"ligand shell ({LIGAND_CUTOFF} A of GAL) .... {sorted(shell)}")
        print(f"UniProt ACT_SITE / BINDING ............ {sorted(UNIPROT_SITES)}")
        print(f"union ................................. {sorted(derived)}")
        print(f"matches the committed constant ........ {derived == ACTIVE_SITE}\n")
        if derived != ACTIVE_SITE:
            raise SystemExit("derived active site differs from ACTIVE_SITE — update the constant")

    act = load(args.activity, "mutation", "act_minus_DGJ")
    foldx = load(args.foldx, "mutation", "ddg")
    scan = {r["mutation"]: r for r in csv.DictReader(args.scan.open())}
    pos = lambda m: int(MUT_RE.match("p." + m).group(2))

    mature = [m for m in act if 32 <= pos(m) <= 429]     # the scanned mature chain
    shared = sorted(m for m in mature if m in scan and m in foldx)
    print(f"Lukas 2013 missense variants ............ {len(act)}")
    print(f"  in the mature chain (32-429) .......... {len(mature)}")
    print(f"  also in the FoldX table ............... {sum(m in foldx for m in mature)}")
    print(f"  also scored by the Boltz scan ......... {len(shared)}\n")
    if not shared:
        print("no overlap — nothing to test", file=sys.stderr)
        return 1

    a = np.array([act[m] for m in shared])
    b = np.array([float(scan[m][args.ddg_col]) for m in shared])
    f = np.array([foldx[m] for m in shared])
    keep = np.array([pos(m) not in ACTIVE_SITE for m in shared])

    print("Rank correlation with residual activity (negative = correct direction)")
    print(f"{'':38}{'n':>5}{'rho':>9}{'p':>9}")
    for name, x, sel in ((f"Boltz {args.ddg_col}", b, slice(None)),
                         ("Boltz, no active-site residues", b, keep),
                         ("FoldX, same mutations", f, slice(None)),
                         ("FoldX, no active-site residues", f, keep)):
        rho, p = spearman(x[sel], a[sel])
        print(f"  {name:36}{len(a[sel]):5}{rho:+9.3f}{p:9.3f}")
    lo, hi = boot_rho(b, a)
    print(f"  Boltz rho 95% CI ................... [{lo:+.3f}, {hi:+.3f}]")

    # FoldX on the Lukas variants the scan has not reached: shows whether the
    # overlap is a representative subset or a harder one.
    rest = [m for m in mature if m in foldx and m not in scan]
    rho_rest, _ = spearman([foldx[m] for m in rest], [act[m] for m in rest])
    rho_all, _ = spearman([foldx[m] for m in mature if m in foldx],
                          [act[m] for m in mature if m in foldx])
    print(f"\n  FoldX on all {len(mature):3} Lukas variants ..... rho = {rho_all:+.3f}")
    print(f"  FoldX on the {len(rest):3} not yet scored ...... rho = {rho_rest:+.3f}")

    # Paired bootstrap: is Boltz different from FoldX on the mutations both cover?
    print()
    for label, sel in (("all", np.ones(len(a), bool)), ("no active-site", keep)):
        obs, lo_d, hi_d, better = paired_boot(b[sel], f[sel], a[sel])
        print(f"  paired rho(Boltz) - rho(FoldX), {label:15} (n={sel.sum():2}) {obs:+.3f} "
              f"CI [{lo_d:+.3f}, {hi_d:+.3f}]  P(Boltz better) = {better:.2f}")

    # Dead (class I, 0% activity) vs any residual activity.
    for label, sel in (("all", np.ones(len(a), bool)), ("no active-site", keep)):
        aa, dead = a[sel], a[sel] == 0
        print(f"\n  [{label}] dead (0% activity, n={dead.sum()}) vs alive (n={(~dead).sum()}), "
              f"median predicted ddG")
        for name, x in (("Boltz", b[sel]), ("FoldX", f[sel])):
            pv = stats.mannwhitneyu(x[dead], x[~dead], alternative="greater").pvalue
            print(f"    {name:6}{np.median(x[dead]):+7.2f} vs {np.median(x[~dead]):+7.2f}"
                  f"   MWU p = {pv:.3f}")

    with_dgj = {r["mutation"]: r["act_plus_DGJ"] for r in csv.DictReader(args.activity.open())}
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        extra = [c for c in ("ddg_A_tsuboyama", "ddg_B_fireprot", "ddg_D_finetuned")
                 if c in next(iter(scan.values()))]
        w.writerow(["mutation", "position", "activity_pct_wt", "activity_pct_wt_plus_DGJ",
                    "active_site", "boltz"] + extra + ["foldx"])
        for m in shared:
            r = scan[m]
            w.writerow([m, pos(m), act[m], with_dgj.get(m, ""),
                        int(pos(m) in ACTIVE_SITE), r[args.ddg_col]]
                       + [r[c] for c in extra] + [foldx[m]])
    print(f"\nwrote {args.out.name} ({len(shared)} rows)")

    plot(a[keep], b[keep], f[keep], args.figure)
    print(f"wrote {args.figure.relative_to(HERE)}")
    return 0


def plot(a, b, f, out: Path) -> None:
    """Scatter of predicted ddG vs measured activity, active-site variants removed.

    They are dropped rather than marked because at those positions activity is not
    a stability proxy at all: the variant can be catalytically dead while folded.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, x, name, symlog in ((axes[0], b, "Boltz-2 embedding scan", False),
                                (axes[1], f, "FoldX", True)):
        ax.scatter(x, a, s=40, c="#2b6cb0", alpha=.85, edgecolor="white", linewidth=.6)
        # Trend within terciles of the prediction. Mean, not median: 24 of the 41
        # variants sit at exactly 0 % activity, so the median is 0 in most bins and
        # hides the signal. No least-squares line either — the statistic reported is
        # a rank correlation, and on FoldX an OLS slope would be set by the handful
        # of clash-regime values. The label under each bar gives the zero-inflation.
        edges = np.quantile(x, [0, 1 / 3, 2 / 3, 1.0])
        for k in range(3):
            m = (x >= edges[k]) & ((x <= edges[k + 1]) if k == 2 else (x < edges[k + 1]))
            if not m.any():
                continue
            ax.hlines(a[m].mean(), edges[k], edges[k + 1], color="#c53030", lw=2.4, zorder=3)
            ax.text(np.sqrt(edges[k] * edges[k + 1]) if symlog and edges[k] > 0
                    else (edges[k] + edges[k + 1]) / 2, a[m].mean() + 6,
                    f"{int((a[m] == 0).sum())}/{int(m.sum())} dead", ha="center",
                    fontsize=8, color="#c53030")
        rho, p = spearman(x, a)
        ax.set_title(f"{name}\nρ = {rho:+.3f}   (p = {p:.3f})", fontsize=11)
        ax.set_xlabel("predicted ΔΔG  [kcal/mol]")
        if symlog:
            ax.set_xscale("symlog", linthresh=1)
        ax.grid(alpha=.25, lw=.6)
    axes[0].set_ylabel("residual α-Gal A activity  [% of wild type]")
    fig.suptitle(f"Predicted destabilisation vs measured residual activity "
                 f"(n = {len(a)} variants, active-site positions excluded)", fontsize=12)
    axes[0].plot([], [], color="#c53030", lw=2.2, label="mean activity per tercile of prediction")
    axes[0].legend(fontsize=8.5, loc="upper right", framealpha=.9)
    fig.tight_layout()
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    raise SystemExit(main())
