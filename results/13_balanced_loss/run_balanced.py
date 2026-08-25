"""13 — Balanced-MSE reweighting of the stabilizing tail.

results/12 found the model's one non-artifactual deficit: stabilizing mutations
(4.3 % of Tsuboyama; bias +0.56, rho 0.27, MAE 2x the class's own spread).
Constraint-aware SPURS (arXiv 2606.08100) reports S669 rho 0.486 -> 0.540 from
loss-level changes alone, of which Balanced MSE is the term aimed at this tail.

Losses compared (identical architecture, data, splits, seeds):
  mse   : plain MSE                                        (baseline == results/06 setup)
  bmc   : Balanced MSE, Monte-Carlo form (Ren et al. CVPR 2022) -- the SPURS "BMC" term
  lds   : inverse-density sample weighting (label smoothing over a ddG histogram)

Evaluated on held-out Tsuboyama (5-fold GroupKFold on wt_id) and, for the same
trained models, transferred to S669. Reports overall AND stabilizing-subset metrics,
because a tail method must not be judged by pooled correlation alone.

    conda run -n ddG_with_Boltz python results/13_balanced_loss/run_balanced.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
Z, N_SEED, N_FOLD = 128, 2, 5
FEAT = [f"{p}_{j}" for p in ("wtz", "mtz") for j in range(Z)]
STAB = -0.5          # ddG < -0.5 kcal/mol == experimentally stabilizing (Mutate Everything)
torch.set_num_threads(8)


# ---------------------------------------------------------------- data
def mat(df):
    return df[FEAT].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)


def augment(X, y):
    """antisymmetry: swap [wtz|mtz] halves, negate ddg (project default, results/07)."""
    Xa = np.concatenate([X[:, Z:], X[:, :Z]], axis=1)
    return np.vstack([X, Xa]), np.concatenate([y, -y])


# ---------------------------------------------------------------- losses
class BMCLoss(nn.Module):
    """Balanced MSE, Monte-Carlo form (Ren et al., CVPR 2022).

    Treats the batch as a classification over which target each prediction belongs to;
    this implicitly divides out the training label density, so rare labels stop being
    dominated by the bulk. noise_var is learned in log space.
    """

    def __init__(self, init_noise=1.0):
        super().__init__()
        self.log_noise = nn.Parameter(torch.tensor(float(np.log(init_noise))))

    def forward(self, pred, target):
        noise_var = self.log_noise.exp()
        logits = -0.5 * (pred - target.T).pow(2) / noise_var
        labels = torch.arange(pred.shape[0], device=pred.device)
        return F.cross_entropy(logits, labels) * (2 * noise_var).detach()


def lds_weights(y, n_bins=40, sigma=2.0):
    """Inverse smoothed-density weights, normalised to mean 1."""
    edges = np.linspace(y.min(), y.max(), n_bins + 1)
    idx = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
    hist = np.bincount(idx, minlength=n_bins).astype(float)
    # gaussian smoothing of the empirical label histogram
    k = np.arange(-3 * int(sigma), 3 * int(sigma) + 1)
    kern = np.exp(-0.5 * (k / sigma) ** 2)
    kern /= kern.sum()
    sm = np.convolve(hist, kern, mode="same")
    w = 1.0 / np.maximum(sm[idx], 1e-6)
    return (w / w.mean()).astype(np.float32)


# ---------------------------------------------------------------- model
def make_net(d_in):
    return nn.Sequential(
        nn.Linear(d_in, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )


def fit(X, y, loss_kind, seed, epochs=60, bs=512, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed)
    net = make_net(X.shape[1])
    crit = BMCLoss().to("cpu") if loss_kind == "bmc" else None
    params = list(net.parameters()) + (list(crit.parameters()) if crit else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    w = lds_weights(y) if loss_kind == "lds" else np.ones(len(y), np.float32)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    wt = torch.from_numpy(w).unsqueeze(1)

    n = len(Xt)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            if len(b) < 8:
                continue
            xb, yb, wb = Xt[b], yt[b], wt[b]
            pred = net(xb)
            if loss_kind == "bmc":
                loss = crit(pred, yb)
            elif loss_kind == "lds":
                loss = (wb * (pred - yb) ** 2).mean()
            else:
                loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    return net


def predict(nets, X):
    with torch.no_grad():
        return np.mean([n(torch.from_numpy(X)).squeeze(1).numpy() for n in nets], axis=0)


# ---------------------------------------------------------------- metrics
def ndcg_stab(y, pred, k=30):
    """nDCG over the stabilizing ranking: gain = max(0, -ddG)."""
    gain = np.maximum(0.0, -y)
    order = np.argsort(pred)                # most-stabilizing prediction (lowest ddG) first
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gain[order][:k] * disc).sum()
    idcg = (np.sort(gain)[::-1][:k] * disc).sum()
    return float(dcg / idcg) if idcg > 0 else np.nan


def metrics(y, pred, tag):
    stab = y < STAB
    m = {
        "set": tag, "n": int(len(y)), "n_stab": int(stab.sum()),
        "r": float(np.corrcoef(y, pred)[0, 1]),
        "rho": float(spearmanr(y, pred).statistic),
        "mae": float(np.abs(pred - y).mean()),
        "stab_bias": float((pred[stab] - y[stab]).mean()) if stab.any() else np.nan,
        "stab_rho": float(spearmanr(y[stab], pred[stab]).statistic) if stab.sum() > 5 else np.nan,
        "stab_mae": float(np.abs(pred[stab] - y[stab]).mean()) if stab.any() else np.nan,
        "auc_stab": float(roc_auc_score(stab, -pred)) if 0 < stab.sum() < len(y) else np.nan,
        "detpr30": float(stab[np.argsort(pred)[:30]].mean()),
        "ndcg30": ndcg_stab(y, pred),
    }
    return m


def main():
    tsu = pd.read_parquet(ROOT / "data/processed/tsuboyama_bench_fast/features_ablation.parquet")
    X, y = mat(tsu), tsu["ddg"].to_numpy(float)
    groups = tsu["wt_id"].to_numpy()
    print(f"Tsuboyama {len(tsu)} muts / {tsu.wt_id.nunique()} proteins; "
          f"stabilizing (ddG < {STAB}): {(y < STAB).sum()} ({(y < STAB).mean():.1%})")

    s669 = pd.read_parquet(ROOT / "data/processed/s669/features_ablation.parquet")
    Xs = mat(s669)
    ys = -s669["ddg"].to_numpy(float)      # -> positive = destabilizing, as in results/12
    print(f"S669 {len(s669)} variants; stabilizing: {(ys < STAB).sum()}\n")

    rows, oof_store = [], {}
    for kind in ("mse", "bmc", "lds"):
        oof = np.full(len(y), np.nan)
        s669_pred = []
        for fi, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLD).split(X, y, groups), 1):
            Xa, ya = augment(X[tr], y[tr])
            imp = SimpleImputer(strategy="median").fit(Xa)
            sca = StandardScaler().fit(imp.transform(Xa))
            T = lambda Q: sca.transform(imp.transform(Q)).astype(np.float32)
            nets = [fit(T(Xa), ya, kind, seed) for seed in range(N_SEED)]
            oof[te] = predict(nets, T(X[te]))
            s669_pred.append(predict(nets, T(Xs)))
            print(f"  [{kind}] fold {fi}/{N_FOLD} r={np.corrcoef(y[te], oof[te])[0,1]:.3f}",
                  flush=True)
        oof_store[kind] = oof
        m = metrics(y, oof, "tsuboyama_oof"); m["loss"] = kind; rows.append(m)
        ps = np.mean(s669_pred, axis=0)
        if np.corrcoef(ys, ps)[0, 1] < 0:
            ps = -ps
        m = metrics(ys, ps, "s669_transfer"); m["loss"] = kind; rows.append(m)
        print(f"  [{kind}] done\n", flush=True)

    res = pd.DataFrame(rows)[["loss", "set", "n", "n_stab", "r", "rho", "mae",
                              "stab_rho", "stab_bias", "stab_mae", "auc_stab",
                              "detpr30", "ndcg30"]]
    res.to_csv(OUT / "results.csv", index=False)
    pd.DataFrame({**{f"pred_{k}": v for k, v in oof_store.items()},
                  "y": y, "wt_id": groups}).to_csv(
        ROOT / "data/processed/_analysis/balanced_oof.csv", index=False)

    for s in ("tsuboyama_oof", "s669_transfer"):
        print(f"=== {s} ===")
        print(res[res.set == s].drop(columns=["set"]).round(3).to_string(index=False))
        print()
    print(f"wrote {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
