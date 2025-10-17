import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import logging
from scipy.stats import pearsonr, skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class ResultAnalyzer:
    """
    Analyzes the results of the feature extraction pipeline, generating PCA plots
    from raw diff tensors and correlation plots from the feature summary.
    """
    def __init__(self, config: dict):
        fe_config = config['feature_extraction']
        out_dir = fe_config['boltz_flags']['out_dir']

        # Todo se toma desde el mismo directorio base
        self.results_dir = out_dir
        self.summary_csv_path = os.path.join(out_dir, "features_summary.csv")

        # Directorio de salida para gráficos
        self.plots_output_dir = os.path.join(out_dir, "exploration_plots")
        os.makedirs(self.plots_output_dir, exist_ok=True)

    @staticmethod
    def _featurize_tensor(tensor):
        """Converts a tensor into a fixed-size feature vector using statistical descriptors."""
        if tensor is None or tensor.size == 0:
            return np.array([np.nan] * 9)
        flat_tensor = tensor.flatten()
        return np.array([
            np.mean(flat_tensor), np.std(flat_tensor), np.median(flat_tensor),
            np.min(flat_tensor), np.max(flat_tensor), np.percentile(flat_tensor, 25),
            np.percentile(flat_tensor, 75), skew(flat_tensor), kurtosis(flat_tensor)
        ])

    def run_global_pca_analysis(self, embedding_types=("pdistogram", "s", "z")):
        """
        Loads diff tensors, vectorizes them, and runs PCA for each embedding type.
        """
        if not os.path.exists(self.summary_csv_path):
            logging.error(f"Summary file not found at '{self.summary_csv_path}'. Cannot run PCA.")
            return

        df_summary = pd.read_csv(self.summary_csv_path)
        df_summary['folder_name'] = df_summary['sequence_id'] + '_' + df_summary['mutation']
        logging.info(f"Starting PCA analysis for {len(df_summary)} mutations.")

        for emb_type in embedding_types:
            logging.info(f"--- Performing PCA for '{emb_type}' tensors ---")

            features_list = []
            for _, row in df_summary.iterrows():
                file_path = os.path.join(self.results_dir, row['folder_name'], f"diff_{emb_type}.npz")
                tensor = np.load(file_path)['arr_0'] if os.path.exists(file_path) else None
                features_list.append(self._featurize_tensor(tensor))

            feature_matrix = np.array(features_list)
            valid_mask = ~np.isnan(feature_matrix).any(axis=1)

            if np.sum(valid_mask) < 2:
                logging.warning(f"Not enough valid data for '{emb_type}' PCA after filtering.")
                continue

            self._plot_pca_results(
                data_matrix=feature_matrix[valid_mask],
                labels=df_summary.loc[valid_mask, 'folder_name'].to_list(),
                colors=df_summary.loc[valid_mask, 'ddg'].to_list(),
                markers=df_summary.loc[valid_mask, 'sequence_id'].to_list(),
                title=f"Tensor '{emb_type}'"
            )

    def _plot_pca_results(self, data_matrix, labels, colors, markers, title):
        """Performs PCA and generates a scatter plot."""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_matrix)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)

        df_plot = pd.DataFrame({
            'PC1': principal_components[:, 0],
            'PC2': principal_components[:, 1],
            'label': labels,
            'ddg': colors,
            'protein': markers
        })

        plt.figure(figsize=(14, 10))
        sns.scatterplot(
            data=df_plot, x='PC1', y='PC2',
            hue='ddg', style='protein',
            palette='coolwarm', s=80
        )

        plt.title(f'Global PCA - {title}', fontsize=16)
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

        output_path = os.path.join(self.plots_output_dir, f"pca_global_{title.replace(' ', '_').lower()}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"PCA plot saved to: {output_path}")

    def run_feature_correlation_analysis(self):
        """
        Generates a multi-panel scatter plot of feature correlations with ddG.
        """
        if not os.path.exists(self.summary_csv_path):
            logging.error(f"Summary file not found at '{self.summary_csv_path}'. Cannot run correlation analysis.")
            return

        df = pd.read_csv(self.summary_csv_path)
        feature_columns = [col for col in df.columns if col.startswith('delta_')]
        if not feature_columns:
            logging.warning("No feature columns (starting with 'delta_') found in the summary file.")
            return

        num_features = len(feature_columns)
        grid_size = math.ceil(math.sqrt(num_features))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 5))
        axes = axes.flatten()
        logging.info(f"Generating correlation plots for {num_features} features...")

        for i, feature in enumerate(feature_columns):
            ax = axes[i]
            temp_df = df[['ddg', feature]].dropna()
            corr, _ = pearsonr(temp_df['ddg'], temp_df[feature]) if len(temp_df) > 1 else (float('nan'), 0)

            sns.regplot(data=df, x='ddg', y=feature, ax=ax,
                        scatter_kws={'alpha': 0.6, 's': 15},
                        line_kws={'color': 'red'})
            ax.set_title(feature, fontsize=10, weight='bold')
            ax.set_xlabel("Experimental ddG", fontsize=8)
            ax.set_ylabel(feature, fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.text(0.05, 0.95, f'$R = {corr:.2f}$', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='aliceblue', alpha=0.7))

        for j in range(num_features, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Feature Correlation with Experimental ddG", fontsize=16, y=0.97)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = os.path.join(self.plots_output_dir, "feature_correlation_plot.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        logging.info(f"Correlation plot saved to: {output_path}")
