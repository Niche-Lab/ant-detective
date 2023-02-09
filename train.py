# native imports
import argparse

# local imports
from models.model import load_model
from data.dataset import create_loader
from train_wrapper import train_wrapper

# pytorch
import torch.nn as nn
import torch.optim as optim


def main(argv: list) -> None:
    model = load_model(model="ViT", dir_weights=argv.dir_weights)
    loader = create_loader(dataname=argv.name_data)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=argv.lr)
    train_wrapper(model, loader, criterion, optimizer, num_epochs=argv.num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_weights", type=str, default="model.pt")
    parser.add_argument("--name_data", type=str, default="peptone_sucrose")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    argv = parser.parse_args()

    main(argv)
