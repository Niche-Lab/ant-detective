# custom modules
from models.model import freeze_param, load_model
from data.dataset import create_loader

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim


def wrapper(argv: list) -> None:
    model = load_model(model="ViT", dir_weights=argv.dir_weights)
    loader = create_loader(dir_data=argv.dir_data)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
