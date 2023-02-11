# native imports
import argparse
import os
import gc

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# local imports
from models.model import load_model, get_device
from data.dataset import create_loader
from wrapper import train_wrapper, test_wrapper
from misc import Timer


def main(argv: list) -> None:
    # display config
    print("Config:")
    for k, v in vars(argv).items():
        print(f"\t{k}: {v}")

    # CONSTANTS
    ROOT = os.path.dirname(os.path.abspath(__file__))
    PATH_MODEL = os.path.join(ROOT, "models")
    PATH_OUT = os.path.join(ROOT, "out")

    # run
    timer = Timer()
    find_ants(
        weights=argv.weights,
        data=argv.data,
        epochs=argv.epochs,
        batch=argv.batch,
        lr=argv.lr,
        inference=argv.inference,
        demo=argv.demo,
        path_model=PATH_MODEL,
        path_out=PATH_OUT,
    )
    timer.report()


def find_ants(
    weights: str,
    data: str,
    epochs: int,
    batch: int,
    lr: float,
    inference: bool,
    demo: bool,
    path_model: str,
    path_out: str,
    delta: float = 5.0,
) -> float:

    # config
    weights = os.path.join(path_model, weights) if weights else None
    model = load_model(model="ViT", dir_weights=weights)
    if demo:
        data += "_demo"
        epochs = 1
    loader = create_loader(name_data=data, batch_size=batch)
    criterion = nn.HuberLoss(delta=delta)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = get_device()

    # train
    if not inference:
        model = train_wrapper(
            model=model,
            loaders=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=epochs,
            path_out=path_model,
        )
    # inference
    loss = test_wrapper(
        model=model,
        loaders=loader,
        criterion=criterion,
        device=device,
        path_out=path_out,
    )
    # gc
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # return
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--data", type=str, default="peptone_sucrose")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--demo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--inference", action=argparse.BooleanOptionalAction, default=False
    )
    argv = parser.parse_args()
    main(argv)
