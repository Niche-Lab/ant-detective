# torch imports
import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

# local imports
from .niche import Niche_Model


class Niche_ViT(Niche_Model):
    def __init__(self):
        super(Niche_ViT, self).__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        self.model = vit_b_16(weights=weights)
        self.model.heads = nn.Linear(self.model.heads.head.in_features, 512)
        self.mlps = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 2),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.mlps(x)
        return x
