# 17 — Does the transfer gap depend on the backbone? Six frozen trunks, one readout

**Status: planned.** No results yet. The phased plan, the pre-registered endpoint and
the running log are in [`status.md`](status.md).

**What:** run the same frozen-trunk ΔΔG pipeline on **six** structure-prediction
backbones instead of one, holding the corpus, the features (`zdiag`, 128 d), the
readout (MLP) and the splits fixed. Train on Tsuboyama, select on FireProt, confirm
on S669.

**Why:** the project's open problem is **transfer**, not in-distribution fit. Random
CV reaches r 0.78 and the homology holdout 0.765 (`01`, `06`), but S669 leakage-free
sits at ρ 0.500 (`16`) and results/11 showed the missing term is **domain shift**,
not a per-protein property. Every attempt to close it from the *readout* side has
failed — biophysical features (`14`), loss reweighting (`13`), calibration offsets
(`11`), fine-tuning (`08`). The one input never varied is the **backbone itself**.
This experiment varies it.

Three questions the single-backbone design cannot answer:

1. **Is the signal a property of the structure-prediction objective, or of Boltz-2?**
   Boltz-2, Protenix-v1, Chai-1 and OpenFold-3 are all AF3-class with a 128-d pair
   track, independently trained on different data cuts. If `zdiag` transfer holds
   across all four, that is a much stronger claim than the project currently makes,
   and it is the direct test of point 3 in `theory/sota_2026.md`.
2. **How much of it is the MSA?** ESMFold is a single-sequence backbone from a
   different lineage entirely. `04` measured the MSA contribution as an *input*
   ablation (~0.08–0.10 r); this measures it as an *architectural* one.
3. **Do independent trunks fail on the same mutations?** If not, cross-backbone
   disagreement is a label-free confidence signal for the stabilizing tail that
   `02`/`12`/`13` could not rank. This costs no GPU — it is a join over feature
   tables Phase 2 already produces.

**How:** the predict step is the only backbone-specific stage. Everything below it
(`slim` → `features` → `ddg.evaluation.transfer`) is already backbone-agnostic: it
consumes `<raw_features_dir>/predictions/<key>/embeddings_<key>.npz` holding
`s`, `z`, `pdistogram`. Each new arm is a runner that writes that NPZ. All six
candidate backbones inherit AF2's **128-wide pair track**, so `zdiag` stays 128-d
and the feature builder, `TRANSFER_BLOCKS` and the readouts run unmodified.

## The arms

| arm | license (code / weights) | `s` × `z` | upstream patch needed | MSA |
|---|---|---|---|---|
| **Boltz-2** (incumbent) | MIT / MIT | 384 × 128 | already done (`external/boltz_modified`) | ColabFold a3m |
| **ESMFold** | MIT / MIT | 1024 × 128 | **none** — `s_s`, `s_z`, `distogram_logits` are output fields | none (single-seq) |
| **OpenFold / AF2** | Apache-2.0 / CC-BY-4.0 | 384 × 128 | **none** — `outputs["pair"]`, `outputs["single"]` | ColabFold a3m |
| **Protenix-v1** | Apache-2.0 / Apache-2.0 | 384 × 128 | small — `get_pairformer_output` | ColabFold-compatible |
| **Chai-1** | Apache-2.0 / Apache-2.0 | AF3 conv. (read at runtime) | small — trunk locals | a3m → `aligned.pqt` |
| **OpenFold-3** | Apache-2.0 / Apache-2.0 | AF3 conv. (read at runtime) | small — trunk return | ColabFold pipeline |

Excluded: RoseTTAFold2 (a third architecture family, but unmaintained since Apr 2025),
OmegaFold (dead since 2022), DeepMind AlphaFold3 (application-gated non-commercial
weights).

Upstream patch points, verified 2026-09-07:
`protenix/model/protenix.py:503` · `chai_lab/chai1.py:746-778` ·
`openfold3/projects/of3_all_atom/model.py:325` · `openfold/model/model.py:450`.

## Data & provenance

| Item | Path / name |
|---|---|
| Train corpus | `tsuboyama_bench_fast` — 12,359 mutations / 412 proteins, 12,772 structures |
| Screen corpus (Phase 1) | `fireprot_le200` — 1,543 mutations / 85 proteins |
| **Selection corpus** | FireProt ≤500 homology-filtered — 130 proteins (`fireprot_le200` + `fireprot_201to500`) |
| **Confirmation corpus** | S669 leakage-free — 411 variants / 67 proteins (`s669`) |
| Configs | `experiment_configs/<corpus>_<backbone>.yaml` (to be written) |
| Processed | `data/processed/<corpus>_<backbone>/` |
| Feature tables | `data/processed/<corpus>_<backbone>/features_summary.parquet` |
| Eval entry point | `python -m ddg.evaluation.transfer --train … --test … --model mlp` (submit it — never the login node) |
| Paired bootstrap | reuse `results/16_aftoolkit_headtohead/headtohead.py::paired_bootstrap` |
| Leakage audit | `results/16_aftoolkit_headtohead/domain_leakage_audit.py` |

## Files
- `status.md` — the plan, the pre-registration, the cluster operating rules, and
  the running log.
- `validate_contract.py` — the Phase 0 gate: checks a backbone's raw embeddings
  against the NPZ contract and that `Δz[i,i]` actually responds to the mutation.
