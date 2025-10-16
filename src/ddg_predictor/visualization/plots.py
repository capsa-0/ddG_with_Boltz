import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import logging
from scipy.stats import pearsonr # <--- 1. Importar la función necesaria

def plot_feature_correlations(summary_csv_path: str, output_dir: str):
    """
    Generates a multi-panel scatter plot showing the correlation between each calculated
    feature and the experimental ddG values, displaying the Pearson correlation coefficient
    on each subplot.
    """
    if not os.path.exists(summary_csv_path):
        logging.error(f"Summary file not found at: {summary_csv_path}")
        return

    df = pd.read_csv(summary_csv_path)
    
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
        
        # --- SECCIÓN AÑADIDA PARA CALCULAR LA CORRELACIÓN ---
        # 2. Eliminar filas con valores NaN para un cálculo correcto
        temp_df = df[['ddg', feature]].dropna()
        
        # 3. Calcular el coeficiente de correlación de Pearson (R)
        if len(temp_df) > 1:
            corr, _ = pearsonr(temp_df['ddg'], temp_df[feature])
        else:
            corr = float('nan') # Asignar NaN si no hay suficientes datos
        # ----------------------------------------------------

        sns.regplot(
            data=df,
            x='ddg',
            y=feature,
            ax=ax,
            scatter_kws={'alpha':0.6, 's':15},
            line_kws={'color':'red', 'linewidth':2}
        )
        
        ax.set_title(feature, fontsize=10, weight='bold')
        ax.set_xlabel("Experimental ddG", fontsize=8)
        ax.set_ylabel(feature, fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)

        # --- SECCIÓN AÑADIDA PARA MOSTRAR EL TEXTO ---
        # 4. Añadir el valor de R al gráfico
        ax.text(0.05, 0.95, f'$R = {corr:.2f}$',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='aliceblue', alpha=0.7))
        # ----------------------------------------------

    # Oculta los subplots no utilizados
    for j in range(num_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Correlation with Experimental ddG", fontsize=16, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = os.path.join(output_dir, "feature_correlation_plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    
    logging.info(f"Correlation plot saved to: {output_path}")