"""
Critical Evaluation Metrics for Risk Models.

Accuracy frequently masks unsatisfactory performance in highly imbalanced classes
(e.g., the actual incidence of fraud in payments). This module enforces robust
metrics tailored for the financial domain.
"""

import torch
import torch.nn.functional as F
from typing import Dict

def calculate_auc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (AUC).
    
    In native PyTorch without external libraries, we use a simplified concordant 
    pairs approach. For production, `torchmetrics.AUROC` is highly recommended.
    
    Args:
        logits (Tensor): Predictions (before Sigmoid).
        targets (Tensor): Ground truth (0 or 1).
    """
    with torch.no_grad():
        preds = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        
        pos_preds = preds[targets == 1]
        neg_preds = preds[targets == 0]
        
        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return 0.5
            
        # Matrix of positive x negative pairs (Warning: O(N*M) memory)
        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)
        concordant = (diff > 0).float().sum()
        ties = (diff == 0).float().sum() * 0.5
        
        auc = (concordant + ties) / (len(pos_preds) * len(neg_preds))
        return auc.item()

def calculate_f1_flexible(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculates F1-Score with a flexible threshold.
    
    Crucial for acute imbalance (e.g., Corporate Bankruptcy in V4FinBench), where
    the threshold must be tuned to prioritize Recall over Precision depending on
    business risk appetite.
    """
    with torch.no_grad():
        preds = (torch.sigmoid(logits).view(-1) >= threshold).float()
        targets = targets.view(-1).float()
        
        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1.item()

def calculate_imputation_rmse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    """
    RMSE (Root Mean Square Error) for Missing Data Inference (Imputation).
    
    Evaluates the network's resilience to missing data patterns occurring routinely
    in degraded transactional systems.
    
    Args:
        preds (Tensor): Network predicted values.
        targets (Tensor): Original ground truth values.
        mask (Tensor): Boolean mask indicating where the value was artificially missing.
    """
    with torch.no_grad():
        if mask.sum() == 0:
            return 0.0
            
        masked_preds = preds[mask]
        masked_targets = targets[mask]
        
        mse = F.mse_loss(masked_preds, masked_targets)
        rmse = torch.sqrt(mse)
        return rmse.item()

class RiskMetricsEvaluator:
    """Aggregator of metrics for validation and testing loops."""
    @staticmethod
    def evaluate(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
        return {
            "auc": calculate_auc(logits, targets),
            "f1_score": calculate_f1_flexible(logits, targets, threshold)
        }
