# torch imports
import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

# local imports
from niche import Niche_Lightning


class Niche_ViT_TWO_NUMBERS(Niche_Lightning):
    def __init__(
        self,
        # base
        lr=1e-3,
        optimizer="Adam",
        loss="MSE",
    ):
        # init
        super().__init__(loss=loss, optimizer=optimizer, lr=lr)

        # backbone ViT_b_16
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        self.vit = vit_b_16(weights=weights)
        self.vit.heads = nn.Identity()

        # MLP 1
        self.mlp1 = nn.Sequential(
            # 1st layer
            nn.Linear(self.vit.heads.head.in_features, 512),
            nn.GELU(),
            nn.Dropout(0.01),
            # 2nd layer
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.01),
            # 3rd layer
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.01),
            # 4th layer
            nn.Linear(128, 1),
        )
        # MLP 2
        self.mlp2 = nn.Sequential(
            # 1st layer
            nn.Linear(self.vit.heads.head.in_features, 512),
            nn.GELU(),
            nn.Dropout(0.01),
            # 2nd layer
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.01),
            # 3rd layer
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.01),
            # 4th layer
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.vit(x)
        out1 = self.mlp1(x)
        out2 = self.mlp2(x)
        # concatenate to one tensor
        return torch.cat((out1, out2), dim=1)


class Niche_ViT(Niche_Lightning):
    def __init__(
        self,
        # base
        lr=1e-3,
        optimizer="Adam",
        loss="MSE",
    ):
        # init
        super().__init__(loss=loss, optimizer=optimizer, lr=lr)

        # backbone ViT_b_16
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        self.vit = vit_b_16(weights=weights)
        self.vit.heads = nn.Sequential(
            # 1st layer
            nn.Linear(self.vit.heads.head.in_features, 512),
            nn.GELU(),
            nn.Dropout(0.01),
            # 2nd layer
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.01),
            # 3rd layer
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.01),
            # 4th layer
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.vit(x)
        return x
