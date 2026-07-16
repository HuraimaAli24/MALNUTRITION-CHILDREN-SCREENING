"""
Multi-task deep learning model for child malnutrition risk prediction.

Architecture:
    Shared trunk (learns joint representation from demographic/household features)
    -> 3 task-specific heads: stunting (HAZ), wasting (WHZ), underweight (WAZ)
    -> each head outputs a continuous z-score regression AND a 3-class severity logit

MC Dropout (dropout kept active at inference) is used to produce predictive
uncertainty by running multiple stochastic forward passes.
"""

import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """One task head: regression (z-score) + classification (severity: normal/moderate/severe)."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Linear(hidden_dim, 1)          # predicted z-score
        self.classifier = nn.Linear(hidden_dim, 3)          # normal / moderate / severe

    def forward(self, x):
        h = self.body(x)
        z = self.regressor(h).squeeze(-1)
        logits = self.classifier(h)
        return z, logits


class MalnutritionMultiTaskNet(nn.Module):
    """
    Shared-trunk multi-task network for stunting, wasting, and underweight prediction.

    Args:
        n_numeric: number of continuous numeric features (age, mother_age, wealth_index...)
        n_region: number of unique region categories (for embedding)
        n_residence: number of unique residence-type categories (urban/rural, small so one-hot is fine)
        embed_dim: embedding size for region
        dropout: dropout probability used throughout (kept active at inference for MC Dropout)
    """

    def __init__(self, n_numeric: int, n_region: int, embed_dim: int = 8, dropout: float = 0.25):
        super().__init__()
        self.region_embed = nn.Embedding(n_region, embed_dim)
        self.dropout_p = dropout

        trunk_in = n_numeric + embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.stunting_head = TaskHead(64, dropout=dropout)      # HAZ
        self.wasting_head = TaskHead(64, dropout=dropout)       # WHZ
        self.underweight_head = TaskHead(64, dropout=dropout)   # WAZ

    def forward(self, x_numeric: torch.Tensor, region_idx: torch.Tensor):
        region_vec = self.region_embed(region_idx)
        x = torch.cat([x_numeric, region_vec], dim=1)
        shared = self.trunk(x)

        haz_z, haz_logits = self.stunting_head(shared)
        whz_z, whz_logits = self.wasting_head(shared)
        waz_z, waz_logits = self.underweight_head(shared)

        return {
            "haz_z": haz_z, "haz_logits": haz_logits,
            "whz_z": whz_z, "whz_logits": whz_logits,
            "waz_z": waz_z, "waz_logits": waz_logits,
        }

    def enable_mc_dropout(self):
        """Keep dropout layers active during inference for MC Dropout uncertainty estimation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
