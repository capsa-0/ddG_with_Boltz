# src/ddg_predictor/models/predictor.py

import torch
import torch.nn as nn

class DDGPredictor(nn.Module):
    def __init__(self, pdist_channels=64, s_channels=384, z_channels=128, mlp_hidden_features=256):
        super().__init__()

        # --- Rama 1: CNN 2D para 'pdistogram' (NxNx64) ---
        self.pdist_encoder = nn.Sequential(
            nn.Conv2d(in_channels=pdist_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Pooling global para manejar tamaño variable N
        )
        
        # --- Rama 2: CNN 1D para 's' (Nx384) ---
        self.s_encoder = nn.Sequential(
            nn.Conv1d(in_channels=s_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Pooling global
        )
        
        # --- Rama 3: CNN 2D para 'z' (NxNx128) ---
        self.z_encoder = nn.Sequential(
            nn.Conv2d(in_channels=z_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # --- Cabezal de Predicción (MLP) ---
        # Sumamos las características de salida de cada rama (32 + 32 + 64)
        combined_input_features = 32 + 32 + 64
        self.predictor_head = nn.Sequential(
            nn.Linear(combined_input_features, mlp_hidden_features),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden_features, 1) # Salida escalar para ddG
        )

    def forward(self, pdistogram, s, z):
        # PyTorch espera los tensores en formato (Batch, Canales, N, ...)
        # Así que permutamos las dimensiones.
        pdistogram = pdistogram.permute(0, 3, 1, 2) # (B, N, N, C) -> (B, C, N, N)
        s = s.permute(0, 2, 1)            # (B, N, C) -> (B, C, N)
        z = z.permute(0, 3, 1, 2)            # (B, N, N, C) -> (B, C, N, N)
        
        # 1. Procesa cada tensor por su rama especializada
        pdist_features = torch.flatten(self.pdist_encoder(pdistogram), 1)
        s_features = torch.flatten(self.s_encoder(s), 1)
        z_features = torch.flatten(self.z_encoder(z), 1)
        
        # 2. Concatena las características obtenidas de cada rama
        combined = torch.cat([pdist_features, s_features, z_features], dim=1)
        
        # 3. Realiza la predicción final
        prediction = self.predictor_head(combined)
        
        return prediction