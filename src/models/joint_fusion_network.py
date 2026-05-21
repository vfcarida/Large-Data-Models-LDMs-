"""
Joint Fusion Network (End-to-End Fusion Layer) using DCNv2.

Enables explicit polynomial interaction between dense Transformer embeddings 
and classical tabular heuristic features. Crucial for binary classification 
tasks where foundation models must respect legacy business rules.
"""

import torch
import torch.nn as nn
from typing import Tuple

class CrossNetworkV2(nn.Module):
    """
    Mathematical implementation of a Deep & Cross Network v2 (DCNv2).
    
    Unlike standard Multi-Layer Perceptrons (MLPs) that learn implicit interactions,
    DCNv2 learns explicit bounded-degree feature interactions. It limits the exponential 
    growth of parameters while capturing high-order correlations between the 
    neural representation and raw tabular inputs.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # Projection matrices for interactions (W and b)
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim)) 
            for _ in range(num_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) 
            for _ in range(num_layers)
        ])
        
        # Xavier initialization for numerical stability
        for w in self.cross_weights:
            nn.init.xavier_uniform_(w)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Calculates feature interactions for the Cross Network V2.
        Formula: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
        """
        xl = x0
        for i in range(self.num_layers):
            # Linear projection W_l * x_l
            proj = torch.matmul(xl, self.cross_weights[i]) + self.cross_biases[i]
            # Explicit interaction with the original input + residual connection
            xl = x0 * proj + xl
        return xl

class EndToEndFusionLayer(nn.Module):
    """
    Final adaptation layer for Binary Classification tasks (e.g., Credit Risk).
    
    Inputs:
    1. Contextual embedding resulting from the last hidden layer of the Transformer.
    2. Tensor of classical tabular features (e.g., bureau score, macroeconomic indicators).
    """
    def __init__(self, transformer_hidden_size: int, tabular_feature_size: int, dcn_layers: int = 3):
        super().__init__()
        
        self.combined_dim = transformer_hidden_size + tabular_feature_size
        
        # Polynomial interaction network
        self.cross_net = CrossNetworkV2(input_dim=self.combined_dim, num_layers=dcn_layers)
        
        # Parallel Deep Network for non-linear dense representations
        self.deep_net = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim // 2),
            nn.BatchNorm1d(self.combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.combined_dim // 2, self.combined_dim // 4),
            nn.BatchNorm1d(self.combined_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final projection combining Cross and Deep outputs
        self.final_projection = nn.Linear(self.combined_dim + (self.combined_dim // 4), 1)

    def forward(self, transformer_emb: torch.Tensor, tabular_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the fusion layer.
        
        Args:
            transformer_emb (Tensor): [Batch, HiddenSize] (Usually from <CLS> token or Pooling).
            tabular_features (Tensor): [Batch, TabularDim] Continuous features from legacy systems.
            
        Returns:
            torch.Tensor: [Batch, 1] Binary logits (Without Sigmoid, suited for BCEWithLogitsLoss).
        """
        # Projective Concatenation
        x = torch.cat([transformer_emb, tabular_features], dim=-1)
        
        # DCNv2: Polynomial interaction
        cross_out = self.cross_net(x)
        
        # Deep Network: Dense non-linear transformations
        deep_out = self.deep_net(x)
        
        # Final Fusion
        fused = torch.cat([cross_out, deep_out], dim=-1)
        logits = self.final_projection(fused)
        
        return logits
