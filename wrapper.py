# torch imports
import torch
import torch.nn as nn

# native imports
import copy
import os
import pandas as pd

# local imports
from models.model import get_device
from misc import Timer, BatchCounter


def wrapper(
    model: nn.Module,
    loaders: dict,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10,
    mode: str = "train",  # train or test
) -> None:

    # config
    ROOT = os.path.dirname(os.path.abspath(__file__))
    PATH_MODEL = os.path.join(ROOT, "models")
    PATH_OUT = os.path.join(ROOT, "out")
    timer = Timer()
    device = get_device()
    model.float().to(device)
    # model.check_param()

    if mode == "train":
        train_wrapper(
            model, loaders, criterion, optimizer, device, PATH_MODEL, num_epochs
        )
    elif mode == "test":
        test_wrapper(model, loaders, criterion, optimizer, device, PATH_OUT)

    timer.report()


def train_wrapper(
    model: nn.Module,
    loaders: dict,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    path_out: str,
) -> None:

    model.float().to(device)

    # metrics
    val_acc_history = []
    train_acc_history = []
    best_loss = 10e10

    # epochs
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # loader
            dataloader = loaders[phase]
            counter = BatchCounter(num_batches=len(dataloader))
            running_loss = 0
            for inputs, labels in dataloader:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # print which batch is being processed
                loss = loss.item()
                counter.report(loss=loss)
                running_loss += loss * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            print("")
            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
            if phase == "val":
                val_acc_history.append(epoch_loss)
                torch.save(
                    model.state_dict(),
                    os.path.join(path_out, "ViT_%d_%.3f.pt" % (epoch, epoch_loss)),
                )
            elif phase == "train":
                train_acc_history.append(epoch_loss)

        print()

    history = dict({"train": train_acc_history, "val": val_acc_history})
    plot_curve(history, name=os.path.join(path_out, "loss.png"))
    print("Best val loss: {:4f}".format(best_loss))


def test_wrapper(
    model: nn.Module,
    loaders: dict,
    criterion: nn.Module,
    device: torch.device,
    path_out: str,
) -> None:

    model.float().to(device)
    model.eval()
    dataloader = loaders["test"]
    counter = BatchCounter(num_batches=len(dataloader))
    pred = []
    running_loss = 0
    for inputs, labels in dataloader:
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            pred.append(outputs.cpu().detach().numpy())

        # print which batch is being processed
        loss = loss.item()
        counter.report(loss=loss)
        running_loss += loss * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print("")
    print("{} Loss: {:.4f}".format("test", epoch_loss))

    # save prediction
    matrix_out = pred[0]
    df_pred = pd.DataFrame(matrix_out)
    df_pred.to_csv(os.path.join(path_out, "pred.out"), index=False)


def plot_curve(history: dict, name: str = "loss.png") -> None:
    import matplotlib.pyplot as plt

    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.legend()
    plt.savefig(name)
    plt.close()
