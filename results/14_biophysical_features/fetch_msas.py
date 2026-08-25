"""14 (item 3) — Fetch WT MSAs from the ColabFold MMseqs2 server.

The alignments this experiment needs were deleted when the corpora were slimmed
(CLAUDE.md: "msas/queries/exploration_plots deleted"); the only a3m files left on
disk belong to the `no_msa` ablation and are single-sequence placeholders. This
refetches them for the **wild-type sequences only** (412 Tsuboyama + 62 S669), which
is all the conservation features need — the MSA is used to compute per-column
statistics, not to re-run Boltz.

Mirrors `MsaGenerator.generate_msas_for_multifasta`: batched submission, one a3m per
`{wt_id}.a3m`, and resumable (an existing a3m is skipped), so a rate-limit failure
partway only costs the unfetched remainder. MSAs are keyed by wt_id and are identical
across corpora sharing the same WT proteins, so this is a one-off cost.

    python results/14_biophysical_features/fetch_msas.py [--exp NAME ...] [--batch 20]
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from external.mmseqs import _run_mmseqs2

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
SCRATCH = Path("/tmp/claude-1000/-media-capsa-Programas-ddG-with-Boltz/"
               "2776204d-251b-4d80-83a6-e933aa1fb33c/scratchpad/msa_tmp")


def wt_sequences(exp: str) -> dict:
    """{wt_id: sequence} for an experiment, from its fasta or its mutations.csv."""
    proc = ROOT / "data/processed" / exp
    fasta = proc / "wt_sequences.fasta"
    if fasta.exists():
        return {r.id: str(r.seq) for r in SeqIO.parse(fasta, "fasta")}
    # S669's processed dir has no fasta — rebuild it from the mutation table
    muts = pd.read_csv(proc / "mutations.csv")
    seqs = muts.groupby("wt_id")["sequence_wt"].first().to_dict()
    with open(fasta, "w") as fh:
        for k, v in seqs.items():
            fh.write(f">{k}\n{v}\n")
    print(f"  [{exp}] wrote {fasta} ({len(seqs)} sequences)")
    return seqs


def fetch(exp: str, batch: int) -> tuple[int, int]:
    out_dir = ROOT / "data/processed" / exp / "msas"
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = wt_sequences(exp)
    pending = {k: v for k, v in seqs.items()
               if not (out_dir / f"{k}.a3m").exists()}
    print(f"[{exp}] {len(seqs)} WT sequences, {len(pending)} to fetch "
          f"-> {out_dir}", flush=True)
    if not pending:
        return 0, 0

    ids = list(pending)
    done, failed = 0, 0
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        t0 = time.time()
        try:
            a3ms = _run_mmseqs2(
                x=[pending[i] for i in chunk],
                prefix=str(SCRATCH / f"{exp}_{start:05d}"),
                use_env=True, use_filter=True,
            )
            for sid, a3m in zip(chunk, a3ms):
                lines = a3m.split("\n")
                if lines and lines[0].startswith(">"):
                    lines[0] = f">{sid}"       # query header = wt_id (pipeline convention)
                (out_dir / f"{sid}.a3m").write_text("\n".join(lines))
            done += len(chunk)
            print(f"  [{exp}] {start + len(chunk)}/{len(ids)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:          # keep what succeeded; a rerun retries the rest
            failed += len(chunk)
            print(f"  [{exp}] batch {start} FAILED: {type(e).__name__}: {e}",
                  flush=True)
    return done, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["tsuboyama_bench_fast", "s669"])
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    os.chdir(SCRATCH)                    # _run_mmseqs2 also drops files in cwd
    total_fail = 0
    for exp in args.exp:
        done, failed = fetch(exp, args.batch)
        total_fail += failed
        print(f"[{exp}] fetched {done}, failed {failed}\n", flush=True)
    if total_fail:
        print(f"{total_fail} sequences still missing — rerun to retry "
              f"(existing a3m files are skipped)")


if __name__ == "__main__":
    main()
