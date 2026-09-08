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

---

## 2026-09-07 — Phase 0 execution started on the cluster

Pushed `25b1589` + `9c3d29f`, pulled on the cluster
(`/grupos/Marce/estructural/ddG_with_Boltz/ddG_with_Boltz`).

### Environment: `transformers` installed, and it moved `click`

`pip install "transformers>=4.44"` in the cluster env resolved to **transformers
5.16.1**, which pulls `typer`; `huggingface-hub 1.30.0` then wants
`click>=8.4.2` and pip upgraded click 8.1.7 → 8.5.0. **`boltz` pins
`click==8.1.7`** — so installing the new arm silently put the *incumbent* arm at
risk.

Resolved by holding boltz's pin and verifying both arms on the cluster:

| check | result |
|---|---|
| `click` version | 8.1.7 (restored) |
| `boltz --help` | OK |
| `from transformers import EsmForProteinFolding` | OK (transformers 5.16.1) |

click is CLI-only for typer/huggingface-hub; neither Python API touches it, so
pip's warning is cosmetic here. `environment.yml` now pins `click==8.1.7`
explicitly with that reasoning, so the next person to rebuild the env does not
"fix" the warning by upgrading click and breaking Boltz.

**ESMFold still exists in transformers 5.x** — checked, since v5 dropped a number
of models. `EsmForProteinFolding` and its `s_s`/`s_z`/`distogram_logits` outputs
are intact.

Note: cluster torch is **2.6.0+cu124** (the workstation has 2.11.0+cu130); the
install did not touch it.

### Two corrections to the runner before submitting

1. **`ensure_esmfold_cache` was going to build the model to warm the cache.** The
   checkpoint is **8.44 GB fp32** (one `pytorch_model.bin`), so `from_pretrained`
   would allocate all of it inside `cpu_step.sbatch`'s 16 GB budget for no reason.
   Now `snapshot_download` — downloads, allocates nothing.
2. **`half_trunk` escape hatch added.** Memory arithmetic for the 8 GB cards:
   8.44 GB fp32 resident, ~5 GB with the language tower in fp16 (`half_esm`),
   leaving ~3 GB for activations. That should hold for Ssym's short chains with
   `chunk_size: 64`. If longer chains OOM, drop `chunk_size` first; `half_trunk`
   is the last resort and **changes the measurement, not just the memory** — the
   runner logs a warning and it must be recorded here if ever enabled.

### Running

- `22288` — `prepare` (cpu). Landed on **nodo3**, one of the known-bad nodes:
  `cpu_step.sbatch` carries no `--exclude` (CLAUDE.md notes the flag is only
  needed for ad-hoc submissions). It survived startup, so nodo3's documented
  `ld.so` failure did not bite this time — but **add
  `--exclude=nodo1,nodo3,nodo4,nodo5` to the CPU submissions too**; there is no
  reason to gamble.
- `no_msa: true` means prepare writes single-sequence a3m files and never
  contacts the MMseqs2 server, then warms the ESMFold cache (8.4 GB download).

### Ssym smoke corpus, as prepared

`prepare` (22288) wrote **350 queries / 350 single-sequence a3m / 337 mutations
over 13 proteins**, then began the ESMFold download (~120 MB/min → ~70 min for
8.44 GB; the weights are cached once and every later arm run reuses them).

**Caveat this corpus cannot address:** Ssym's longest chain is **164 aa**
(`2LZMA`), so a clean pass here validates the contract and the basic memory
settings but says **nothing about the 8 GB length ceiling**. results/16 had to
measure Boltz's ceiling explicitly with a length ladder (job 20745: 701 aa ✓,
795 aa ✗). ESMFold needs the same probe before Phase 1, because
`fireprot_le200` goes to 200 aa and `s669` to 500. Do not infer a ceiling from
Ssym.

Predict submission for this corpus: 350 structures over 16 shards ≈ 22 each.

### Length ladder built (prerequisite for Phase 1)

`data/raw/esmfold_length_probe.csv` + `experiment_configs/esmfold_length_probe.yaml`
— 12 rungs, one variant per protein:

| aa | 149 | 201 | 261 | 297 | 345 | 419 | 448 | 505 | 619 | 701 | 795 | 1207 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

The lower rungs matter more than the upper ones here. results/16's existing
`s669_long_probe` starts at 505 aa, which was the right range for Boltz; ESMFold
carries ~5 GB of resident weights on an 8 GB card, so its ceiling may fall
*below* 505 and the existing ladder would not see it. `fireprot_le200` needs
200 aa and `s669` / `fireprot_201to500` need ~500 — a ceiling under 500 means
the ESMFold arm cannot cover the selection corpus without `half_trunk`.

`slurm/probe_length.sbatch` needed no change: it runs `ddg run --step predict`,
which now dispatches on `backbone`, and it already reports GPU model, compute
capability and peak VRAM per rung with one query per array task.

### Fixed: the CPU wrappers were not excluding the bad nodes

Only `predict_array.sbatch` carried `--exclude=nodo1,nodo3,nodo4,nodo5`; prepare
22288 was scheduled onto **nodo3** as a result. Added the list to
`cpu_step.sbatch` and `probe_length.sbatch` (CLAUDE.md notes it is harmless on
CPU steps). Commit `304a973`.

### 22290 — predict failed on an OOM in my own load order (fixed, `20d0257`)

The dispatch worked: the shard logged `Running esmfold predictions...` then
`Running ESMFold on shard 0/16 (22 of 350 queries)`, i.e. the registry resolved
and the ESMFold runner took over from Boltz. It then died in `_load_model`:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB.
GPU 0 has a total capacity of 10.90 GiB of which 10.12 MiB is free.
... torch/nn/modules/module.py line 1329, in convert -> return t.to(
```

**Cause was mine, not the hardware.** `_load_model` did `model.to(device)` and
*then* `model.esm.half()`, so the whole **fp32 checkpoint had to fit on the card
during the copy** and `half_esm` never ran. It OOM'd on nodo10's **11 GB** card,
which is larger than the 8 GB the memory plan was written around — so this says
nothing yet about whether ESMFold fits once the cast happens first.

Fixed by casting before the move. The runner now also logs the resident weight
size and free VRAM after load, so the memory question is answered by the log
rather than inferred from a crash.

**Note the GPU inventory is not uniform.** `sinfo -p gpu` shows nodo10 with a
~11 GB card; results/16 measured its length ladder on an 8 GB RTX 2080, and
results/09 hit a ~1.94 GiB card on nodo12. Any VRAM number from Phase 0 must be
recorded **with the node and card it came from**, or the ceiling is meaningless.

Resubmitted as **22294** (16 shards, `%2`).

### 22294 — it runs, and the memory picture is worse than planned

First real numbers, from shard 1 on **nodo10**:

```
ESMFold on cuda (half_esm=True, half_trunk=False, chunk_size=64)
    — 7.86 GiB of weights resident
GPU NVIDIA GeForce GTX 1080 Ti: 2.68 GiB free of 10.90 GiB after load
```

**`half_esm` is a no-op.** Two independent confirmations:

1. Resident weights (7.86 GiB) **equal the on-disk checkpoint** (8.44 GB =
   7.86 GiB). If the tower had been fp32 on disk, halving it would have cut
   resident well below the file size. It did not, so it was already fp16.
2. The arithmetic only closes that way. `config.json` says
   `esm_type: esm2_3B` — 2.8 B params, which is **11.2 GB in fp32, larger than
   the entire 8.44 GB checkpoint**. At fp16 the tower is ~5.2 GiB, and the trunk
   (48 blocks, seq 1024, pair 128, ~0.7 B params fp32) is ~2.6 GiB: 7.8 GiB
   total, matching the measurement.

So the checkpoint ships the language tower in fp16 regardless of the config's
`fp16_esm: False`, and the flag I added to control it does nothing. **The only
remaining memory lever is `half_trunk`** (~2.6 GiB → ~1.3 GiB, so ~6.5 GiB
resident) — and that one changes the measurement, not just the memory.

**Consequence for scheduling: this arm cannot run on the 8 GB cards at all.**
7.86 GiB of weights plus the CUDA context does not fit in 8 GB before a single
activation is allocated. results/16 measured Boltz's ladder on an 8 GB RTX 2080;
ESMFold needs the ≥11 GB nodes, or `half_trunk`. The GPU fleet is **not
uniform** — GTX 1080 Ti (10.9 GiB, Pascal cc 6.1) here, RTX 2080 (8 GB, Turing
cc 7.5) in results/16, ~1.94 GiB on nodo12 in results/09 — so **every VRAM
number in this experiment must be recorded with its node and card**.

**2.68 GiB free after load is the activation budget**, and it is what will set
the length ceiling. Ssym's 164 aa chains fit. The pair track scales as L², so
500 aa (needed for `s669` and `fireprot_201to500`) is ~9× the activation
footprint — the ladder is now the decisive Phase 0 measurement, not a formality.

Also noted: `esm_type: esm2_3B`, trunk `num_blocks: 48`, `sequence_state_dim:
1024`, `pairwise_state_dim: 128` — confirming the 128-wide pair track the whole
comparison depends on, read from the model's own config rather than assumed.

### The GPU fleet, measured (this belongs in CLAUDE.md eventually)

`srun --gres=gpu:1 -w <node> nvidia-smi --query-gpu=name,memory.total,compute_cap`:

| node | GPU | VRAM | compute cap | holds ESMFold (7.86 GiB)? |
|---|---|---|---|---|
| nodo6 | RTX 2080 | 8192 MiB | 7.5 | **no — OOM** |
| nodo7 | RTX 2080 | 8192 MiB | 7.5 | **no** |
| nodo8 | RTX 2080 | 8192 MiB | 7.5 | **no** |
| nodo10 | GTX 1080 Ti | 11264 MiB | 6.1 | **yes** (2.68 GiB free after load) |
| nodo11 | RTX 3050 | 6144 MiB | 8.6 | **no** |

Confirmed by the array rather than inferred: **22294_1 on nodo10 COMPLETED
22/22; 22294_0 and 22294_2 on nodo6 FAILED** with
`CUDA out of memory ... total capacity of 7.60 GiB`.

**So the ESMFold arm currently has one usable GPU node.** That is a scheduling
constraint, not a blocker — see the throughput below — but every ESMFold
submission must carry `-w nodo10` (or an exclude list that leaves only the
≥11 GB nodes), or most shards will fail.

### Throughput: the budget claim holds

Shard 1, 22 structures at ≤164 aa on nodo10:

```
12:46:45  shard starts
12:47:31  model loaded          ->  46 s one-time load
12:49:39  22/22 predictions     ->  128 s  =  5.8 s/structure
```

(The 16:52 `sacct` elapsed is the whole array task — predict *plus* the
per-shard slim — not the predict cost. Do not read throughput off `Elapsed`.)

Against the Boltz fit `t ≈ 2.3 + 4e-4·L²`, Boltz at 164 aa would be ~13 s, so
**ESMFold is ~2× faster than Boltz at the same length** and the Phase 1 budget's
"ESMFold well under the Boltz unit" survives. On one node, Phase 1's ESMFold
share (~2,900 structures) is ~4.7 GPU-h ≈ 5 h wall, and a full Tsuboyama pass
(12,772) is ~20 h. Workable serialized.

**Shard count should go DOWN for this arm, not up.** The 46 s model load is paid
per shard, so CLAUDE.md's "many short shards" rule — written for Boltz, where
startup is ~3 min against 65 s/structure — inverts here: 16 shards on one node
costs ~12 min of pure loading for 34 min of work. Use ~4.

### `half_trunk` is riskier than "changes the measurement"

It would drop resident weights to ~6.5 GiB, fitting the three RTX 2080s — which
are also *Turing*, with fast fp16, unlike nodo10's Pascal card. Tempting.

But the feature this project reads is a **difference**, `zdiag = mut_z[i,i] −
wt_z[i,i]`, and fp16 carries ~3 decimal digits. A small difference between two
larger numbers is exactly the case where fp16 loses the signal to catastrophic
cancellation. So `half_trunk` is not a free scheduling win; if it is ever used,
the arm must be validated against an fp32 run on overlapping structures before
its numbers are allowed into the endpoint.

Keep the trunk in fp32 and live with one node.

### PASSED — ESMFold's pair diagonal carries a local mutation response

`validate_contract.py` on the first 145 structures (contract over 40, locality
over 25 WT/mutant pairs), run under `srun` on the cpu partition:

```
backbone: esmfold   structures on disk: 145
contract: checked 40 structures, z widths seen: [128]
sanity: checked 25 (WT, mutant) pairs
PASSED — embeddings satisfy the contract and the mutation is local
```

**Every one of the 25 pairs responds at the mutated position**, with
`|Δz[i,i]|` at the mutation between **5.5× and 70×** the far-residue median:

| protein | ratio range (n) | far median |
|---|---|---|
| 1BNIA | 5.5× – 15.1× (8) | 5.4 – 21.0 |
| 1L63A | 13.0× – 70.3× (15) | 1.7 – 7.8 |
| 1IOBA | 29.3× (1) | 5.3 |

This is the answer Phase 0 existed to get. Nothing before this point said an
ESMFold pair track carries mutation signal at all — the synthetic control in the
offline self-test only proved the validator could tell the difference. It does,
it is local, and `Dz = 128` holds across every structure checked, so `zdiag` is
dimension-matched to the Boltz-2 arm and the feature builder needs no change.

Note the locality ratio is **protein-dependent** (1L63A's far field moves ~3×
less than 1BNIA's for a comparable response at the mutation). Not a problem for
the contract, but worth remembering if cross-arm disagreement (Phase 3c) is ever
normalised per protein.

**Remaining Phase 0 gate: the length ceiling** (22313, queued behind 22310 on
nodo10). 2.68 GiB free after weights, pair track scaling as L².

### Length ceiling measured: 505 aa ✓ / 619 aa ✗ — the arm covers the whole plan

`slurm/probe_length.sbatch`, one rung per array task, `-w nodo10`
(GTX 1080 Ti, 11264 MiB, cc 6.1). Full table in `compatibility.csv`:

| aa | 149 | 201 | 261 | 297 | 345 | 419 | 448 | **505** | 619 | 701 | 795 | 1207 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| result | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | ✗ | ✗ | ✗ | ✗ |
| peak MiB | 8749 | 9003 | 9319 | 9541 | 9937 | 10745 | 11017 | 10915 | — | — | — | — |

Failures are genuine OOM, not node faults:
`CUDA out of memory ... total capacity of 10.90 GiB of which 136.12 MiB is free`.

**This clears the arm for the full plan.** 505 aa covers `s669` (longest chain
493 aa), `fireprot_201to500` (≤500), `fireprot_le200` and Tsuboyama (~70 aa).
The worry recorded earlier — that a sub-500 ceiling would force the selection
corpus down to `fireprot_le200` and cost the design its statistical power over
S669 — **does not materialise**. Phase 1 stands as pre-registered.

For reference, Boltz's ceiling on the 8 GB RTX 2080 was 701 ✓ / 795 ✗
(results/16). ESMFold's is lower *and* on a bigger card, which is the 7.86 GiB
of resident weights showing up.

**But the margin above ~400 aa is thin and must be respected.** 448 aa peaked at
**11017 MiB of 11264 — 97.8 % of the card**, ~250 MiB spare; 505 aa at 10915.
Two consequences:

1. Long-chain shards are **fragile, not safe**. Anything else touching that GPU
   tips them over. Keep ESMFold's long-chain work to `--gres=gpu:1` on an
   otherwise idle nodo10, and expect occasional OOM on requeue rather than
   treating it as a new bug.
2. If `fireprot_201to500` throws scattered OOMs, the fix is **`chunk_size` 64 →
   32 or 16** (activation memory, and it does not touch the numerics), *not*
   `half_trunk` (which does).

Caveat on the numbers: `probe_length.sbatch` samples `nvidia-smi` every 2 s, so
a short spike between samples is missed — treat these peaks as lower bounds.

### Phase 0 verdict for the ESMFold arm: **PASS**

| gate | result |
|---|---|
| runs on this cluster | yes — **nodo10 only** (7.86 GiB weights; 8 GB cards OOM on load) |
| NPZ contract | yes — `s` (L×1024), `z` (L×L×**128**), `pdistogram`; slim + features unchanged |
| `Δz[i,i]` responds to the mutation | yes — **5.5×–70×** the far-residue median over 25 pairs |
| length ceiling | **505 aa ✓ / 619 aa ✗** — covers every corpus in the plan |
| throughput | **5.8 s/structure** at ≤164 aa, ~2× faster than Boltz at that length |

Cleared for Phase 1. Remaining before the screen runs: build the Tsuboyama 10 %
protein subsample, and write `fireprot_le200_esmfold.yaml` — both with
`delete_raw: true` restored and `-w nodo10`.

---

## 2026-09-07 — hardware probe: two of the six arms are not runnable on this cluster

Static check of the three AF3-class arms against the measured GPU fleet
(largest card **11264 MiB, cc 6.1**; no card has bf16 except nodo11's RTX 3050,
which is 6 GB).

| arm | upstream requirement | verdict |
|---|---|---|
| **Chai-1** | README: "requires ... a GPU with CUDA and **bfloat16** support"; recommends A100/H100/L40S 48–80 GB, minimum A10/A30 (24 GB) | **OUT** — the only bf16 card here is 6 GB, far under even their minimum |
| **OpenFold-3** | Installation.md: "requires ... CUDA 12.1 and **32 GB** of memory"; tested on A100 40 GB | **OUT** — fleet maximum is 11 GB |
| **Protenix** | `dtype: bf16` is a *default*, and `runner/inference.py:218` maps `"fp32": torch.float32`; `triangle_attention` has a pure-`torch` fallback beside `cuequivariance` | **plausible, untested** |

**This changes the pre-registered arm list.** The plan's cleanest scientific
claim was *four independently-trained AF3-class models sharing a 128-d pair
track* — if `zdiag` transfer holds across all four, the signal belongs to the
structure-prediction objective rather than to any one training set. With Chai-1
and OpenFold-3 unrunnable here, that axis has **at most two** members
(Boltz-2 and Protenix), which weakens the claim from "a property of the model
class" to "reproduced in one other model".

Realistic arm list, revised:

| arm | status |
|---|---|
| Boltz-2 | incumbent, done |
| **ESMFold** | **Phase 0 PASSED** |
| OpenFold / AF2 | proven on this cluster — results/16 ran AFToolkit's AF2 pipeline on nodo6/nodo8 |
| Protenix | pivotal and untested; now the *only* remaining AF3-class partner |
| Chai-1 | out (bf16 + VRAM) |
| OpenFold-3 | out (32 GB) |

**Protenix is the arm to fight for**, because without it there is no
same-architecture-class comparison at all. It needs an **isolated conda env**:
it pins `torch==2.7.1` against the cluster env's 2.6.0, and after the `click`
incident an in-place install into `ddG_with_Boltz` is not acceptable — a torch
change would put Boltz, the incumbent, at risk. Plan: `ddg_protenix` env with
`pip install -e .` for `ddg`, then `dtype: fp32` +
`triangle_attention: torch` on nodo10.

Not attempted yet. Recorded as the next Phase 0 item.

**Neither exclusion is a statement about the models** — both are hardware
limits of this cluster, and both would run on any 40 GB card. If GPU time
elsewhere becomes available, they go back on the list unchanged.

---

## 2026-09-07 — Phase 1 launched; the screen instrument is validated

### The screen works: 42 proteins still transfer

Before spending GPU on new arms, the incumbent was run through the screen to ask
whether a 10 % training subsample transfers at all. It does:

| Boltz-2, trained on | test | pooled ρ | pooled r | per-protein median r |
|---|---|---|---|---|
| **42 proteins** (screen) | FireProt ≤200, n=1543 / 85 prot | **0.595** | 0.555 | 0.573 |
| 412 proteins (results/05, ≤500) | FireProt ≤500 | ~0.66 | 0.65 | 0.65 |

A ~0.065 drop for a 10× smaller training set — **exactly what results/03's
learning curve predicts** (33 proteins → r 0.744 vs 330 → 0.793). Training on a
tenth of the corpus does not destroy transfer, so the screen is a usable ranking
instrument, and the Phase 1 design holds.

This also costs no GPU: the Boltz-2 arm is a pure subset of its existing
412-protein table, so the baseline every other arm is measured against is free.

### Corpora prepared and predicting

| job | corpus | structures | role |
|---|---|---|---|
| 22417 | tsuboyama_screen10_esmfold | 1,302 | prepare ✓ |
| 22418 | fireprot_le200_esmfold | 1,628 | prepare ✓ |
| 22419 | tsuboyama_screen10_esmfold | | predict, 6 shards, `-w nodo10` |
| 22420 | fireprot_le200_esmfold | | predict, chained `afterany:22419` |

Chained rather than concurrent: both need the single 11 GB card, and
interleaving would make every task re-pay the 46 s model load.

### Two bugs in the screen runner, both found by running it

1. **The Boltz-2 train table was the wrong file.**
   `tsuboyama_bench_fast/features_summary.parquet` is a **legacy pre-refactor
   table** (657 cols, `local_s_dim_*_signed_diff`) that shares *no* feature
   columns with the raw-Δz tables — `ddg.evaluation.transfer` died with "no
   shared feature columns between train and test tables". The raw-Δz table for
   that corpus is **`rawz_features.parquet`** (`zdiag_*`/`zpool_*`), which is
   also what results/03's provenance table names. Worth knowing generally: on
   `tsuboyama_bench_fast` those two parquets are different generations, not
   different views.
2. `transfer_summary.json` prefixes its pooled metrics (`pooled_spearman`, …),
   so the screen table was silently collecting `None` for every metric while
   still writing a file. Fixed to read the real keys.

`run_screen.py` enforces the two fairness conditions a bare `transfer` call
cannot: identical training proteins across arms, and a test set **intersected on
(wt_id, mutation)** across arms before scoring, so no arm is flattered by a
variant another arm dropped.

---

## 2026-09-07 — Protenix arm: isolated env built; ESMFold Phase 1 blocked on the queue

### `ddg_protenix` env (isolated, verified)

| | |
|---|---|
| python | 3.11.16 |
| protenix | **2.0.0** (pinned; the build asserts the version) |
| torch | 2.7.1+cu126 |
| size | ~6 GB |
| `ddG_with_Boltz` after | torch 2.6.0+cu124, click 8.1.7, `boltz --help` OK — **untouched** |

Isolation was the point: protenix pins `torch==2.7.1` against the incumbent's
2.6.0, and after the `click` episode an in-place install was not acceptable.

### Three failed attempts, and what each one taught

1. **Env build tied to an ssh session** — killed when the *workstation* ran out
   of memory. Long installs must be `sbatch`-detached, not `srun` under ssh.
2. **`--wrap` pointed at `/tmp/build_protenix_env.sh`** — written on the login
   node, but the job runs on a compute node with its own local `/tmp`. Exit 127
   in 0 s. Anything a batch job executes must live on shared storage.
3. **The build "succeeded" while having failed.** `conda create` died on
   `CondaError: Prefix record 'ca-certificates' already exists` (corrupt package
   cache), but the script had `set -x` and **no `set -e`**, so it ran to
   `echo DONE` and exited 0. Worse, the watch grepped for `protenix import OK`
   and matched the `set -x` *trace* of the command rather than its output — a
   fabricated success signal. Fixed with `set -euo pipefail`, an explicit
   `test -x` on the new interpreter, and `RESULT`/`BUILD_OK` markers that only
   real output can produce. **Nothing was contaminated** — `protenix` and
   `torch` were absent from `base`, and the incumbent env verified clean.

### The version trap worth remembering

The first *successful* build installed **protenix 0.5.5**, not the current
model. Cause: the env was created with `python=3.10` (copying the project's
convention), but **protenix >= 1.0.4 declares `requires_python >= 3.11`**. pip
did not error — it silently backtracked to 0.5.5, the last 3.10-compatible
release. An old model would have stood in as "the Protenix arm" without ever
announcing itself. The build now pins the version and asserts it post-install.

### Pair width: the default model is dimension-matched, v2 is not

`configs/configs_model_type.py` defines several model types, and **they do not
share a pair width**:

| model type | `c_z` | dimension-matched to Boltz-2 / ESMFold? |
|---|---|---|
| `protenix_base_default_v1.0.0` (**the inference default**) | **128** | **yes** |
| `protenix-v2` | **256** | no |

Take the **default v1** for the primary arm. The experiment's whole design is
"hold everything fixed except the backbone", and `protenix-v2`'s 256-wide pair
track would hand that arm twice the readout features — confounding backbone
quality with feature dimensionality. results/14 already showed more dimensions
is not automatically better on transfer (256-d constructions did not beat 128-d
`zdiag`), but it would still be an uncontrolled difference. `protenix-v2` is
worth keeping as an optional *capacity* arm, clearly labelled as not matched.

Confirmed present in the installed package: `get_pairformer_output` returning
`s_inputs, s, z`, with `c_s 384` / `c_z 128` — the patch point the survey
identified, intact in the shipped release.

### Still unanswered for this arm (all need a GPU)

- Does `dtype: fp32` actually run on Pascal (nodo10, cc 6.1) / Turing
  (nodo6-8, cc 7.5)? `torch.cuda.get_arch_list()` returned `[]` on the login
  node because no GPU is visible there — it must be re-checked on a GPU node,
  and torch 2.7 may have dropped Pascal.
- Does `triangle_attention` fall back from `cuequivariance` to `torch`?
- Protenix takes **JSON** input, not the Boltz-style query YAML the pipeline
  emits, so `run_protenix.py` needs a converter — real integration work beyond
  the ~10-line dump the survey estimated.

### ESMFold Phase 1

Still `PENDING`. nodo10 is held by another user's array (`jortigosa`, 6 h+),
with two more users queued for GPUs. Nothing to fix — this is the cost of the
single-node pin, and taking more of a shared cluster is not an option.
