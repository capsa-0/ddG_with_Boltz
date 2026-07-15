"""
Module: plots
Description: Visualization functions for feature analysis and exploration.
Generates correlation plots, UMAP projections, and scatter plots.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

logger = logging.getLogger(__name__)


def plot_correlation_summary(df: pd.DataFrame, output_dir: str, features_per_panel: int = 40):
    """
    Generate multi-panel correlation bar plot ordered by importance.
    
    Args:
        df: DataFrame with features and 'ddg' column
        output_dir: Output directory for plot
        features_per_panel: Features per subplot panel
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ----- Select numeric feature columns -----
    exclude_cols = ['mut_id', 'wt_id', 'mutation', 'ddg', 'sample_id']
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    if not feature_cols:
        logger.warning("No numeric features found for correlation")
        return

    logger.info(f"Computing correlations for {len(feature_cols)} features")
    
    # ----- Compute Pearson correlations -----
    correlations = {}
    for col in feature_cols:
        valid_data = df[['ddg', col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_data) > 2:
            correlations[col] = valid_data['ddg'].corr(valid_data[col])
            
    corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation']).dropna()
    corr_df['Abs_Corr'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values(by='Abs_Corr', ascending=False).drop(columns=['Abs_Corr']).reset_index(drop=True)
    
    # ----- Save full correlation values -----
    csv_path = os.path.join(output_dir, "correlation_values_all.csv")
    corr_df.to_csv(csv_path, index=False)
    logger.debug(f"Saved correlations to {csv_path}")
    
    # ----- Multi-panel layout -----
    num_features = len(corr_df)
    num_panels = math.ceil(num_features / features_per_panel)
    
    fig, axes = plt.subplots(1, num_panels, figsize=(8 * num_panels, 14), sharex=True)
    
    # Ensure axes is iterable
    if num_panels == 1:
        axes = [axes]
        
    for i in range(num_panels):
        start_idx = i * features_per_panel
        end_idx = min((i + 1) * features_per_panel, num_features)
        subset = corr_df.iloc[start_idx:end_idx]
        
        sns.barplot(
            data=subset, y='Feature', x='Correlation', hue='Correlation',
            palette='vlag', dodge=False, legend=False, ax=axes[i]
        )
        
        axes[i].axvline(x=0, color='black', linestyle='-', linewidth=1)
        axes[i].set_title(f'Rank {start_idx+1} to {end_idx}', fontsize=14, pad=10)
        axes[i].set_xlabel('Pearson R' if i == num_panels // 2 else '', fontsize=12)
        axes[i].set_ylabel('')
        axes[i].grid(axis='x', linestyle='--', alpha=0.6)
        
        # ----- Add correlation value labels -----
        for index, value in enumerate(subset['Correlation']):
            axes[i].text(value, index, f' {value:.2f}', va='center', fontsize=9,
                         ha='left' if value > 0 else 'right')
            
    plt.suptitle('Feature Correlation with Experimental ddG', fontsize=18, y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "correlation_summary_panels.png")
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved correlation plot to: {out_path}")


def plot_umap(df: pd.DataFrame, output_dir: str):
    """
    Generate UMAP projection of features colored by ddG values.
    
    Args:
        df: DataFrame with features and 'ddg' column
        output_dir: Output directory for plot
    """
    # Optional dependency: skip the UMAP plot (not the whole features step) if
    # umap-learn isn't installed. The features_summary.parquet is the deliverable.
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed; skipping UMAP projection plot")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ----- Select feature columns -----
    feature_cols = [c for c in df.columns if c not in ['wt_id', 'mut_id', 'mutation', 'ddg']]
    X = df[feature_cols].dropna()
    y = df.loc[X.index, 'ddg']
    
    if len(X) < 5:
        logger.warning("Insufficient data for UMAP")
        return

    logger.info("Computing UMAP projection")
    
    # ----- Compute UMAP -----
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(X)
    
    # ----- Generate plot -----
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='coolwarm', s=50, alpha=0.8)
    plt.colorbar(scatter, label='Experimental ddG')
    plt.title('UMAP Projection of Boltz Features')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_path = os.path.join(output_dir, "umap_projection.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved UMAP plot to: {out_path}")


def plot_correlations(df: pd.DataFrame, output_dir: str):
    """
    Generate individual scatter plots for each feature vs ddG.
    
    Args:
        df: DataFrame with features and 'ddg' column
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    feature_cols = [c for c in df.columns if c not in ['wt_id', 'mut_id', 'mutation', 'ddg']]
    logger.info(f"Generating {len(feature_cols)} scatter plots")
    
    for col in feature_cols:
        temp_df = df[['ddg', col]].dropna()
        if temp_df.empty or len(temp_df) < 2:
            continue
        
        corr = temp_df['ddg'].corr(temp_df[col])
        
        plt.figure(figsize=(6, 5))
        sns.regplot(data=temp_df, x='ddg', y=col, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
        plt.title(f"Pearson: R = {corr:.3f}")
        plt.xlabel("ddG (Experimental)")
        plt.ylabel(col)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        out_path = os.path.join(output_dir, f"corr_{col}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()