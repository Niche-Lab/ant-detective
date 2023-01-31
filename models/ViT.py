# torch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.io import read_image
from torchvision.models import ViT_B_16_Weights
from torchvision.transforms import ToTensor

# custom imports
from niche import Niche_Model


class Niche_ViT(Niche_Model):
    def __init__(self):
        super(Niche_ViT, self).__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        self.model = torchvision.models.vit_b_16(weights=weights)
        self.model.heads = nn.Linear(self.model.heads.in_features, 1)

    def forward(self, x):
        return self.model(x)
