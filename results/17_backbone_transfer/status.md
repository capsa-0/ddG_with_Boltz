# 17 — status log

## 2026-09-07 — experiment planned and pre-registered; nothing run yet

Backbone survey done (see the arms table in `README.md`); all patch points and
licenses verified against upstream `main` on 2026-09-07. This entry is the
**pre-registration**: the endpoint below is fixed *before* any arm has a number.

---

### Pre-registered endpoint

**Primary.** Pooled **Spearman ρ** on the **FireProt ≤500 homology-filtered**
corpus (130 proteins), predictor trained on the whole of `tsuboyama_bench_fast`,
features = `TRANSFER_BLOCKS` (`zdiag`, 128 d) only, readout = MLP, trunk frozen.
One number per arm.

**Secondary.** Per-protein median r on the same corpus; pooled Pearson r.

**Confirmation.** S669 leakage-free (411 variants / 67 proteins), **opened once**,
at Phase 4, for three predictors only: Boltz-2 baseline, best single arm, best
combiner.

**Multiplicity.** Holm across the 5 new arms on the primary. The three combiners
(3a/3b/3c below) are declared in advance and are not a search.

**Why selection ≠ confirmation.** results/16 put the paired protein-cluster
bootstrap CI on S669-leakage-free at **[−0.017, +0.151] for a +0.054 effect** — at
411 variants that corpus cannot resolve differences below ~0.05 ρ. Selecting the
best of six arms on it would be noise-mining, and would also burn S669 for the
AFToolkit headline in `16`. FireProt ≤500 is 3,205 variants over 130 blind
proteins — ~7× the variants — so it is the corpus that can actually rank six arms.

**Prerequisite.** `16`'s coverage-free domain audit is complete for S669 but was
still in progress for FireProt ("FireProt is being screened the same way",
`16/README.md:90`; `domain_leakage_fireprot.csv` has 29 rows). **Finish that audit
and freeze the selection corpus before Phase 1 results are looked at.** MMseqs2
clustering alone is not sufficient here — that is what hid contamination in 34.7 %
of S669.

**Everything except the backbone is held fixed:** corpus, mutation set, MSA
strategy (`mutate_across_msa`), feature blocks, readout, splits, seeds.

---

### Cost model

Two measured anchors from this project's own logs: **~6.0 s/structure at ~70 aa**
(`15`: 25,224 structures → 42 GPU-h) and **~65 s/structure at 398 aa** (`10`).
Fitting `t ≈ 2.3 + 4e-4·L²` through them gives, per arm, in Boltz-2 units:

| corpus | structures | est. GPU-h |
|---|---|---|
| tsuboyama_bench_fast (full) | 12,772 | 21 |
| tsuboyama 10 % screen (~41 prot.) | ~1,270 | 2.1 |
| fireprot_le200 | 1,628 | 5 |
| fireprot_201to500 | 1,715 | 24 |
| s669 | 603 | 4.5 |

ESMFold should come in well under the Boltz-2 unit (no MSA, no diffusion path);
AF2 over it — `16` measured AFToolkit at 32.8 s/variant. Treat the table as an
ordering, not a promise.

**`prepare` does not re-run.** MSAs are already on disk for every corpus, keyed by
`wt_id`, including the mutated ones — and Protenix, OpenFold-3 and Chai-1 all
consume ColabFold a3m (Chai via its `aligned.pqt` converter). Zero MMseqs2 server
load; the marginal cost is pure GPU.

---

### Phase 0 — hardware gate + contract validation  (~5 GPU-h total)

**This is the highest-value step and it runs first.** The cluster GPUs are
**RTX 2080 8 GB, compute capability 7.5** (Turing — no bf16), and
`predict_array.sbatch` already excludes the ~1.94 GiB nodes. Boltz force-disables
its own triangle kernels there; AF2 is **proven** on nodo6/nodo8 (`16` ran
AFToolkit's AF2 pipeline on exactly those cards). The AF3-class arms —
Protenix, Chai-1, OpenFold-3 — default to bf16 and sm_80 kernels and are the real
risk. Chai-1 additionally ships its trunk as exported `.pt` components, so its
precision assumptions are baked in.

Per arm, on one ~70 aa and one ~400 aa sequence:
1. does it run on cc 7.5 / 8 GB at all (fp32 or fp16 fallback)?
2. peak VRAM vs the length ladder in `16` (505 aa ✓ … 795 aa ✗ for Boltz);
3. the NPZ contract: `s` (L×Ds), `z` (L×L×Dz), `pdistogram`, no NaNs, `Dz == 128`;
4. sanity: `z[i,i]` for a mutant differs from WT at the mutated position and is
   ~unchanged far from it.

**Gate:** an arm that needs sm_80, or blows 8 GB below ~500 aa, is dropped here or
moved to an fp32 fallback — before any GPU-hours are committed to it.

**Deliverable:** `compatibility.csv` + a log entry. Cheap, and it can invalidate
up to three arms.

### Phase 1 — screen  (~36 GPU-h, ~18 h wall at `%2`)

5 new arms × [tsuboyama 10 % subsample (2.1) + `fireprot_le200` (5)] ≈ 7 GPU-h/arm,
~36 GPU-h total. At the `%2` courtesy cap that is ~18 h wall, not the ~12 h an
earlier draft assumed at `%3` — CLAUDE.md's rule is many short shards at `%2`.

Justified by `03`: **33 training proteins already reach r 0.744**, 330 reach 0.793,
and the subsample seed SD is **≤0.002** — an order of magnitude below the 0.05–0.10
backbone spreads reported in `theory/sota_2026.md` (AF2 0.56 / MSA-Transformer 0.53
/ ESM-2 0.47). The screen is adequately powered to *rank*.

**Known limitation, stated up front:** the screen assumes the arms do not differ in
**data efficiency**. An arm that loses at 41 proteins could win at 412. The
concavity of `03`'s curve makes a rank flip unlikely but does not exclude it, and
this screen produces rankings, not publishable absolutes.

**Gate:** carry forward every arm within **0.03 ρ** of the best, plus Boltz-2
unconditionally.

### Phase 2 — full training corpus for survivors  (21 GPU-h/arm, + 24 to complete ≤500)

Full `tsuboyama_bench_fast`, then `fireprot_201to500` to complete the ≤500
selection view. Expect 2–3 survivors. ESMFold runs in full regardless of its screen
rank — it is cheap and it is the only single-sequence arm, so its number is
informative either way.

### Phase 3 — combine  (**zero GPU** — joins over Phase 2 feature tables)

This is where a transfer *improvement* would actually come from. Three
pre-declared combiners:

- **3a — prediction-level ensemble.** Average the per-arm predictions. Safest:
  it does not multiply dimensionality and is robust to the per-corpus offset that
  `14` showed level-carrying features import.
- **3b — feature concat.** 6 × 128 = 768-d namespaced `zdiag`
  (`<backbone>__zdiag_<j>`), joined on `(wt_id, mutation)`. Riskier — correlated
  arms may re-import the `11` domain-shift offset rather than cancel it.
- **3c — disagreement as selective prediction.** Cross-arm variance as a
  label-free confidence signal. Test specifically whether the stabilizing-tail
  failures of `02`/`12`/`13` concentrate where the arms disagree. If they do, the
  tail becomes *flaggable* even though `13` showed it is not rankable — and no
  single backbone can give this.

3a and 3b may well lose to the best single arm. That is a publishable negative in
the same line as `11`, `13` and `14`, and it is cheap to establish.

### Phase 4 — confirmation  (4.5 GPU-h/arm)

S669 leakage-free, **single look**, three predictors only (Boltz-2 baseline, best
single arm, best combiner). Paired protein-cluster bootstrap, 4,000 resamples,
reusing `headtohead.py::paired_bootstrap`. If a new arm or combiner beats Boltz-2
here, it also updates the AFToolkit comparison in `16`.

---

### Implementation work items

1. **`backbone:` key** in the experiment YAML; dispatch in
   `ddg/feature_extraction/extract_features.py:39` to
   `ddg/feature_extraction/extraction/run_<backbone>.py`. Every runner writes the
   existing contract: `<raw_features_dir>/predictions/<key>/embeddings_<key>.npz`
   with `s`, `z`, `pdistogram`. Nothing below predict changes.
2. **Derive `Z_DIM` from the array** instead of the module constant
   (`ddg/features/build_features.py:44`, `ddg/scan/predict.py:45`). All six arms
   are 128 so nothing breaks today, but ESMFold's `s` is **1024**-d, which the
   `slim.keep_s` path needs to handle (and which makes its slim store ~2.7× bigger
   on that field if `keep_s` is on — default it off for ESMFold).
3. **Column namespacing** for 3b, joined on `(wt_id, mutation)`.
4. **Runners.** ESMFold and OpenFold/AF2 need no upstream patch. Protenix,
   Chai-1 and OpenFold-3 each need a small embeddings-only patch mirroring
   `external/boltz_modified/changes.md`; keep each under
   `external/<name>_modified/` with its own `changes.md`, same as Boltz.
5. **Configs**: `experiment_configs/<corpus>_<backbone>.yaml`. Do **not** set
   `overwrite: true` — it deletes the whole processed dir for that experiment.

### Risks

| risk | mitigation |
|---|---|
| Turing / 8 GB kills the AF3-class arms | Phase 0, ~5 GPU-h, before any commitment |
| Screen mis-ranks via data-efficiency differences | stated as a limitation; 0.03 ρ carry-forward band is deliberately loose |
| Six arms inflate the forking-paths surface | selection/confirmation split + Holm; endpoint fixed above before any result |
| Ensemble does not beat the best single arm | pre-declared as a legitimate negative |
| FireProt domain audit incomplete | finish it before Phase 1 results are read |
| Shared-cluster courtesy (`%2` cap; `10` was cut for this) | phased with kill gates; worst case ~150 GPU-h, not ~280 |

### Next step

Phase 0. Write `run_esmfold.py` first — no upstream patch, no MSA, and it validates
the `backbone:` dispatch and the NPZ contract end-to-end on the cheapest possible
arm before any AF3-class integration work starts.

---

## 2026-09-07 — Phase 0 part 1: backbone dispatch + ESMFold runner written and validated offline

The `backbone:` dispatch and the ESMFold arm are in. **Nothing has run on a GPU
yet** — the model itself is untested (see *Not yet verified* below).

### What landed

| file | change |
|---|---|
| `ddg/feature_extraction/extraction/common.py` | **new** — shard split, `is_done`, `slimmed_keys`, `select_pending`, `read_query_sequence`. Every backbone now shares one resume/shard implementation. |
| `ddg/feature_extraction/extraction/backbones.py` | **new** — name → (run, warm) registry, `boltz1`/`boltz2`/`esmfold`. Imports are deferred so an env with Boltz but no `transformers` still resolves its own backbone. |
| `ddg/feature_extraction/extraction/run_esmfold.py` | **new** — the ESMFold arm. |
| `ddg/feature_extraction/extraction/run_boltz.py` | refactored onto the shared helpers; behaviour unchanged. |
| `ddg/feature_extraction/extract_features.py` | dispatches on `config.backbone` instead of calling Boltz directly. |
| `ddg/pipeline.py` | `prepare` warms the *selected* backbone's cache, not always Boltz's. |
| `ddg/config/config_loader.py` | `config.backbone` (default `boltz2`) + `config.backbone_flags`; `boltz_flags` became `.get` so a non-Boltz config need not carry it. |
| `ddg/features/build_features.py` | pair width now **derived from the array** (`Z.shape[1] // len(blocks)`) instead of the `Z_DIM = 128` constant, and raises if it does not divide evenly. |
| `experiment_configs/ssym_esmfold.yaml` | **new** — the Phase 0 smoke corpus (337 mutations, the smallest on disk). |
| `results/17_backbone_transfer/validate_contract.py` | **new** — the Phase 0 gate (contract + locality sanity). |
| `tests/test_sharding.py` | import follows `shard_files` into `common`. |
| `environment.yml` | `transformers>=4.44` (was absent from the env). |

**No upstream patch was needed for ESMFold**, as the survey predicted:
`EsmForProteinFoldingOutput(**structure)` is built from a dict already carrying
`s_s`, `s_z` and `distogram_logits`, so a plain forward pass yields all three.
They map onto the contract as `s` (L×1024) / `z` (L×L×128) / `pdistogram` (L×L×64).

`keep_s: false` for this arm: the single track is 1024-d rather than 384, and
results/14 established the s-derived block is not what carries transfer, so
storing it would cost ~2.7× the disk for a field the endpoint never reads.

### Verified offline (no GPU, no model weights)

- Dispatch: every existing config still reports `backbone=boltz2` with
  `backbone_flags == boltz_flags`; the ESMFold config resolves its own flags;
  an unknown name fails with a listing of the known ones.
- **Full contract path on synthetic ESMFold-shaped tensors** (L=40, Ds=1024,
  Dz=128): `_write_prediction` → `ddg.storage.slim` → `build_features` produced
  `zrow (1,40,128)`, `pdrow (1,40,64)` and a 512-column feature table
  (4 blocks × 128) with no code changes below predict. **This is the claim the
  whole experiment rests on, and it holds.**
- **Non-128 pair width**: the same path with Dz=64 produced 64 `zdiag_*` columns,
  not a silently misaligned 128 — the derived-width change works.
- **The validator catches a position-blind backbone.** A control whose mutant `z`
  equals its WT `z` fails with "the backbone is not seeing the substitution"
  and a non-zero exit; the realistic case reports the locality ratio (62.8×).
- `tests/test_prepare.py`, `tests/test_sharding.py`, `tests/test_cli_status.py`
  all pass after the refactor.

### Two bugs the self-test found

1. **`np.savez` appends `.npz`** unless the name already ends in it, so the
   atomic-write temp file was written to `<name>.npz.tmp.npz` and the rename
   failed on every structure. Temp is now `.embeddings_<key>.partial.npz` — still
   dot-prefixed, so a leftover stays invisible to `is_done`'s
   `embeddings_*.npz` glob.
2. The locality diagnostic was suppressed when the far-field shift was exactly
   zero, hiding the numbers a Phase 0 judgement call needs. It now always prints,
   distinguishing "far field unchanged" from "nothing moved anywhere".

### Not yet verified — this is the actual Phase 0 gate

Everything above is shape plumbing. **None of it has run ESMFold.** Still open:

- `transformers` is **not installed** in `ddG_with_Boltz` (torch is: 2.11.0+cu130).
  Install it before the smoke run.
- Does ESMFold fit in **8 GB on cc 7.5**? The ESM-2 3B tower is the problem, not
  the trunk. `half_esm: true` + `chunk_size: 64` is the starting point; lower the
  chunk size first if a long chain OOMs. Turing has fp16 tensor cores but no
  bf16 — do not switch this arm to bf16.
- The length ceiling, against `16`'s Boltz ladder (701 aa ✓ / 795 aa ✗).
- Whether the **real** `Δz[i,i]` is local. The synthetic control only proves the
  validator can tell; the question of whether ESMFold's pair diagonal actually
  responds to a substitution is empirical, and it is the one that decides whether
  this arm is worth Phase 1.

### Cluster rules this experiment must follow

Collected from CLAUDE.md so a future session does not have to rediscover them:

- **Never run compute on the login node.** `python`, `boltz`, feature extraction
  and *analysis scripts* all go through `sbatch` / `srun`. Only `git`, `squeue`,
  `scancel`, `sacct`, `cp`, `ls`, `grep`, `tail`, `mkdir` and `ddg status` are
  safe there.
- **Code reaches the cluster by git**, not by copying: commit and push from the
  workstation, `git pull` at
  `/grupos/Marce/estructural/ddG_with_Boltz/ddG_with_Boltz`. Pulling while jobs
  run is safe, so keep changes backward-compatible for pending jobs — the
  `backbone:` default of `boltz2` is what makes this refactor safe to pull
  mid-run.
- **Env activation inside jobs** is `source /home/shared/load-conda` then
  `conda activate ddG_with_Boltz` (admin-mandated 2026-08-24). The wrappers in
  `slurm/` already do this.
- **Many short shards, `%2` GPUs.** Never a monolithic "whole corpus at once"
  step: a long job that dies near the end wastes the work *and* stalls the
  `afterok` chain.
- **Exclude the bad nodes** — `nodo1` (CUDA init), `nodo3` (ld.so), `nodo4`
  (broken `/grupos` mount, fails *on write*), `nodo5` (bad GPU), plus `nodo9` at
  the admins' request. `predict_array.sbatch` already carries the list; ad-hoc
  submissions need `--exclude=` explicitly. Never name `nodo14`/`nodo15` —
  they were decommissioned and sbatch rejects the whole submission.
- **`PENDING (DependencyNeverSatisfied)` is dead, not waiting** — `scancel` it.

### New dependency: `transformers`

`environment.yml` now pins `transformers>=4.44`, but the **cluster env has to be
updated separately** — this checkout is a different machine. On the cluster,
inside a job (not on the login node):

```bash
source /home/shared/load-conda && conda activate ddG_with_Boltz
pip install "transformers>=4.44"
```

`ensure_esmfold_cache` then downloads the ESMFold weights once, serially, from
the `prepare` step — the same race-avoidance `ensure_boltz_cache` exists for.

### Next step — Phase 0 smoke run

1. Commit and push from the workstation; `git pull` on the cluster.
2. Install `transformers` in the cluster env (above).
3. `prepare` needs **no MSA server**: the config sets `no_msa: true`, so
   `msa_generator` writes single-sequence a3m files and never contacts MMseqs2.

```bash
sbatch slurm/cpu_step.sbatch experiment_configs/ssym_esmfold.yaml prepare

# ~370 structures; short shards, %2 GPUs, bad nodes already excluded by the wrapper
sbatch --array=0-15%2 slurm/predict_array.sbatch experiment_configs/ssym_esmfold.yaml 16

# the gate (analysis is compute too -- srun it, don't run it on the login node)
srun --partition=cpu --time=00:20:00 --mem=8G \
    python results/17_backbone_transfer/validate_contract.py \
    --config experiment_configs/ssym_esmfold.yaml --pairs 20
```

**Note `slim.delete_raw: false` in the Phase 0 config.** `predict_array.sbatch`
self-slims each shard and deletes its raw NPZs; the validator reads the *raw*
predictions, because the locality check needs the whole `z[j,j]` diagonal and the
slim store keeps only `z[pos, :, :]`. Ssym is small enough that keeping raw is
free. **Set `delete_raw: true` again in every Phase 1+ config** — that setting is
what kept the ≤500 FireProt run off a ~180 GB peak.

Record the peak VRAM, the s/structure and the locality ratio in
`compatibility.csv`; those three numbers are what the Phase 0 gate decides on.
Read the node out of the log (`srun: error: <node>: task 0`) before blaming the
model — a failure that clusters on one node is the node, not ESMFold.
