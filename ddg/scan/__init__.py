"""
ddg.scan — exhaustive single-protein ΔΔG scans.

Where ddg.evaluation scores the predictor against *labelled* corpora, this package
points it at one protein with **no labels at all** and asks for every possible
single point mutation: L positions x 19 alternative residues.

Two stages, deliberately separate because only the first needs a GPU:

1. **build** (``python -m ddg.scan build``) — turn a sequence into the L*19
   mutation CSV plus an experiment YAML, so the normal pipeline
   (prepare -> predict -> slim -> features) extracts Boltz embeddings for it.
2. **predict** (``python -m ddg.scan predict``) — fit the ΔΔG regressor on
   labelled corpora and apply it to the scan's feature table, emitting a
   per-mutation table, a position x residue matrix, and heatmaps.

See ddg/scan/build.py and ddg/scan/predict.py.
"""

from ddg.scan.mutations import all_point_mutations, AA_ORDER, STANDARD_AA  # noqa: F401
