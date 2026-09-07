"""
Module: backbones
Description: Registry of embedding backbones the predict step can dispatch to.

The experiment YAML picks one with `feature_extraction.backbone` (default
`boltz2`, so every existing config keeps working untouched). A backbone supplies:

  run  : (config, shard=None) -> None   run predictions, write the NPZ contract
  warm : (config) -> None               populate weight caches serially, or None

Adding one means writing `run_<name>.py` that writes
`<raw_features_dir>/predictions/<key>/embeddings_<key>.npz` with `s`, `z` and
(optionally) `pdistogram`, then registering it here. Nothing below predict changes
— see results/17_backbone_transfer.

Imports are deferred to call time: an environment with Boltz but without
`transformers` (or the reverse) must still be able to run its own backbone.
"""

import logging

logger = logging.getLogger(__name__)

DEFAULT_BACKBONE = "boltz2"


def _boltz():
    from ddg.feature_extraction.extraction.run_boltz import (ensure_boltz_cache,
                                                             run_boltz_predictions)
    return run_boltz_predictions, ensure_boltz_cache


def _esmfold():
    from ddg.feature_extraction.extraction.run_esmfold import (
        ensure_esmfold_cache, run_esmfold_predictions)
    return run_esmfold_predictions, ensure_esmfold_cache


# name -> zero-arg loader returning (run_fn, warm_fn)
BACKBONES = {
    "boltz2": _boltz,
    "boltz1": _boltz,     # same CLI, selected via boltz_flags.model
    "esmfold": _esmfold,
}


def _resolve(name: str):
    try:
        loader = BACKBONES[name]
    except KeyError:
        raise ValueError(
            f"unknown backbone '{name}'; known: {', '.join(sorted(BACKBONES))}"
        ) from None
    return loader()


def get_runner(name: str):
    """Return the predict function for a backbone."""
    return _resolve(name)[0]


def get_warmer(name: str):
    """Return the cache-warming function for a backbone, or None."""
    return _resolve(name)[1]
