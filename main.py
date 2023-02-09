# native imports
import argparse
import os

# pytorch
import torch.nn as nn
import torch.optim as optim

# local imports
from models.model import load_model, get_device
from data.dataset import create_loader
from wrapper import train_wrapper, test_wrapper
from misc import Timer


def main(argv: list) -> None:
    # CONSTANTS
    ROOT = os.path.dirname(os.path.abspath(__file__))
    PATH_MODEL = os.path.join(ROOT, "models")
    PATH_OUT = os.path.join(ROOT, "out")

    # config
    weights = os.path.join(PATH_MODEL, argv.weights)
    model = load_model(model="ViT", dir_weights=weights)
    if argv.demo:
        argv.data += "_demo"
    loader = create_loader(name_data=argv.data)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=argv.lr)

    timer = Timer()
    if argv.mode == "train":
        train_wrapper(
            model=model,
            loaders=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=get_device(),
            num_epochs=argv.num_epochs,
            path_out=PATH_MODEL,
        )
    elif argv.mode == "test":
        test_wrapper(
            model=model,
            loaders=loader,
            criterion=criterion,
            device=get_device(),
            path_out=PATH_OUT,
        )
    timer.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="model.pt")
    parser.add_argument("--data", type=str, default="peptone_sucrose")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--demo", action=argparse.BooleanOptionalAction, default=False)
    argv = parser.parse_args()
    # display config
    print("Config:")
    for k, v in vars(argv).items():
        print(f"\t{k}: {v}")

    main(argv)
