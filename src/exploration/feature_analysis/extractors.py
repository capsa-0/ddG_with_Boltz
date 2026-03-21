"""
Module: extractors
Description: Feature extraction functions for analyzing Boltz embeddings.
Computes mathematical features from wild-type and mutant tensors.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import skew, kurtosis, entropy


# ----- STATISTICAL FEATURE CALCULATIONS -----

def _calculate_entropy(arr: np.ndarray) -> float:
    """
    Calculate Shannon entropy of array.
    
    Args:
        arr: Input array
        
    Returns:
        Shannon entropy value
    """
    arr = np.abs(arr).flatten()
    arr_sum = np.sum(arr)
    if arr_sum == 0:
        return 0.0
    probs = arr / arr_sum
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _calculate_gini(arr: np.ndarray) -> float:
    """
    Calculate Gini coefficient (measure of inequality).
    
    Args:
        arr: Input array
        
    Returns:
        Gini coefficient value
    """
    arr = np.abs(arr).flatten()
    if np.sum(arr) == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float(((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr))))


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Calculate Kullback-Leibler divergence between two distributions.
    
    Args:
        p: First probability distribution (torch tensor)
        q: Second probability distribution (torch tensor)
        
    Returns:
        KL divergence value
    """
    p_np = np.abs(p.detach().cpu().numpy().flatten()) + 1e-10
    q_np = np.abs(q.detach().cpu().numpy().flatten()) + 1e-10
    p_np /= np.sum(p_np)
    q_np /= np.sum(q_np)
    return float(entropy(p_np, q_np))


def _get_stats(tensor: torch.Tensor, prefix: str, L: int) -> dict:
    """
    Calculate comprehensive statistics for a tensor.
    
    Args:
        tensor: Input tensor
        prefix: Feature name prefix
        L: Sequence length for normalization
        
    Returns:
        Dictionary with statistical features
    """
    if tensor.numel() == 0:
        return {}
    
    arr = tensor.detach().cpu().numpy().flatten()
    arr_sum = np.sum(arr)
    
    return {
        f"{prefix}_mean": np.mean(arr),
        f"{prefix}_std": np.std(arr),
        f"{prefix}_max": np.max(arr),
        f"{prefix}_sum": arr_sum,
        f"{prefix}_entropy": _calculate_entropy(arr),
        f"{prefix}_gini": _calculate_gini(arr),
        f"{prefix}_skew": float(skew(arr)) if len(arr) > 2 else 0.0,
        f"{prefix}_kurtosis": float(kurtosis(arr)) if len(arr) > 2 else 0.0,
        f"{prefix}_mean_abs": np.mean(np.abs(arr)),
        f"{prefix}_mean_norm": np.mean(arr), 
        f"{prefix}_sum_norm": arr_sum / L if L > 0 else 0.0,
        f"{prefix}_entropy_norm": _calculate_entropy(arr) / np.log(L) if L > 1 else 0.0,
        f"{prefix}_gini_norm": _calculate_gini(arr),
        f"{prefix}_mean_abs_norm": np.mean(np.abs(arr))
    }


# ----- DIFFERENCE COMPUTATION MODES -----

def compute_diff(wt: torch.Tensor, mut: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Compute difference between wild-type and mutant tensors.
    
    Args:
        wt: Wild-type tensor
        mut: Mutant tensor
        mode: "abs" (absolute), "signed" (signed), or "l2" (euclidean)
        
    Returns:
        Difference tensor
        
    Raises:
        ValueError: If mode is not recognized
    """
    if mode == "abs":
        return torch.abs(mut - wt)
    elif mode == "signed":
        return mut - wt
    elif mode == "l2":
        return torch.norm(mut - wt, dim=-1)
    else:
        raise ValueError(f"Unknown diff mode: {mode}")


def extract_features(sample: dict, mut_pos: int, window_size: int = 5) -> dict:
    """
    Extract comprehensive features from Boltz embeddings.
    
    Computes features based on:
    - Local embedding changes at mutation position
    - Neighborhood context around mutation
    - Matrix interactions (z and pdistogram)
    - KL divergence of spatial distributions
    
    Args:
        sample: Dictionary with Boltz tensors (wt_s, mut_s, wt_z, mut_z, etc)
        mut_pos: Zero-indexed position of mutation in sequence
        window_size: Context window size for neighborhood features
        
    Returns:
        Dictionary with extracted features
    """
    features = {}
    L = sample["wt_s"].shape[0]
    
    wt_s_local = sample["wt_s"][mut_pos]
    mut_s_local = sample["mut_s"][mut_pos]
    
    # ----- Extract per-dimension signed differences -----
    diff_signed_local_s = mut_s_local - wt_s_local
    D = diff_signed_local_s.shape[0]
    for d in range(D):
        features[f"local_s_dim_{d}_signed_diff"] = diff_signed_local_s[d].item()

    # ----- SECTION 1: Wild-type context and similarity -----
    features.update(_get_stats(wt_s_local, prefix="wt_local_s", L=L))
    features["local_s_cosine_sim"] = F.cosine_similarity(
        wt_s_local.unsqueeze(0), mut_s_local.unsqueeze(0)
    ).item()

    # ----- SECTION 2: Iterate over all difference modes -----
    for mode in ["abs", "signed", "l2"]:
        # --- Local changes ---
        diff_s_local = compute_diff(wt_s_local, mut_s_local, mode=mode)
        features.update(_get_stats(diff_s_local, prefix=f"local_s_{mode}", L=L))

        # --- Neighborhood context ---
        start = max(0, mut_pos - window_size)
        end = min(L, mut_pos + window_size + 1)
        diff_s_neigh = compute_diff(
            sample["wt_s"][start:end], sample["mut_s"][start:end], mode=mode
        )
        features.update(_get_stats(diff_s_neigh, prefix=f"neigh_{window_size}_s_{mode}", L=L))

        # --- Matrix interactions (z and pdistogram) ---
        for emb_type in ["z", "pdistogram"]:
            if f"wt_{emb_type}" not in sample:
                continue
                
            wt_tensor = sample[f"wt_{emb_type}"]
            mut_tensor = sample[f"mut_{emb_type}"]
            
            # Local diagonal element
            diff_diag = compute_diff(
                wt_tensor[mut_pos, mut_pos, :],
                mut_tensor[mut_pos, mut_pos, :],
                mode=mode
            )
            features.update(_get_stats(diff_diag, prefix=f"local_{emb_type}_{mode}", L=L))

            # Interaction row (all positions interacting with mutation)
            diff_row = compute_diff(
                wt_tensor[mut_pos, :],
                mut_tensor[mut_pos, :],
                mode=mode
            )
            features.update(_get_stats(diff_row, prefix=f"interact_{emb_type}_{mode}", L=L))
            
    # ----- SECTION 3: KL divergence of spatial probability distributions -----
    if "wt_pdistogram" in sample:
        wt_p = sample["wt_pdistogram"]
        mut_p = sample["mut_pdistogram"]
        features["local_pdistogram_kl_div"] = _kl_divergence(
            wt_p[mut_pos, mut_pos, :],
            mut_p[mut_pos, mut_pos, :]
        )
        features["interact_pdistogram_kl_div"] = _kl_divergence(
            wt_p[mut_pos, :],
            mut_p[mut_pos, :]
        )

    return features