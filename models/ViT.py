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
        self.model.heads = nn.Sequential(
            # 1st layer
            nn.LazyLinear(512),
            nn.GELU(),
            nn.LazyBatchNorm1d(),
            nn.Dropout(0.01),
            # 2nd layer
            nn.LazyLinear(512),
            nn.GELU(),
            nn.LazyBatchNorm1d(),
            nn.Dropout(0.01),
            # 3rd layer
            nn.LazyLinear(128),
            nn.GELU(),
            nn.LazyBatchNorm1d(),
            nn.Dropout(0.01),
            # 4th layer
            nn.LazyLinear(2),
            nn.LazyBatchNorm1d(),
        )

    def forward(self, x):
        return self.model(x)
