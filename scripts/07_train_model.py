# scripts/07_train_model.py

import yaml
import torch
import logging
import argparse # <-- Importar argparse
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from ddg_predictor.data.datasets import DDGDataset, custom_collate_fn
from ddg_predictor.models.predictor import DDGPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    """
    Main training and validation loop.
    """
    logging.info(f"--- Training model using data from: {args.results_dir} ---")
    
    # --- 1. Cargar el Dataset Completo desde el directorio de resultados ---
    full_dataset = DDGDataset(results_dir=args.results_dir)
    
    # --- 2. Dividir el Dataset ---
    # (La lógica de GroupShuffleSplit no cambia)
    metadata_df = full_dataset.metadata
    groups = metadata_df['sequence_id']
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, val_indices = next(gss.split(metadata_df, groups=groups))
    
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    logging.info(f"Dataset split: {len(train_subset)} train / {len(val_subset)} val samples.")

    # --- 3. Crear DataLoaders para cada conjunto ---
    train_loader = DataLoader(
        train_subset, 
        batch_size=8,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=8,
        shuffle=False, # No es necesario barajar en validación
        collate_fn=custom_collate_fn
    )
    
    # --- 4. Inicializar Modelo, Pérdida y Optimizador ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    model = DDGPredictor(mlp_hidden_features=256).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # --- 5. Bucle de Entrenamiento y Validación ---
    num_epochs = 10 # Esto eventualmente vendrá del config
    
    for epoch in range(num_epochs):
        # -- Fase de Entrenamiento --
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            if batch is None: continue
            labels = batch.pop("ddg").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            predictions = model(**inputs)
            loss = loss_fn(predictions, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # -- Fase de Validación --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                if batch is None: continue
                labels = batch.pop("ddg").to(device)
                inputs = {k: v.to(device) for k, v in batch.items()}
                
                predictions = model(**inputs)
                loss = loss_fn(predictions, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        logging.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

    logging.info("Training finished.")
    # (Guardar el modelo final)


if __name__ == "__main__":
    # --- Añadir Parseador de Argumentos ---
    parser = argparse.ArgumentParser(description="Train a ddG prediction model from a results directory.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the self-contained results directory from the feature extraction pipeline."
    )
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
        exit(1)