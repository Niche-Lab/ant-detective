# torch imports
import torch
import torch.nn as nn

# custom modules
from models.model import get_device
from misc import Timer, BatchCounter

# native imports
import copy
import os


def train_wrapper(
    model: nn.Module,
    loaders: dict,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10,
) -> None:

    # config
    ROOT = os.path.dirname(os.path.abspath(__file__))
    timer = Timer()
    device = get_device()
    model.float().to(device)
    model.check_param()

    # metrics
    val_acc_history = []
    train_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10e10

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
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
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            print("")
            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_loss)
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        ROOT, "models", "ViT_%d_%.3f.pt" % (epoch, epoch_loss)
                    ),
                )
            elif phase == "train":
                train_acc_history.append(epoch_loss)

        print()

    history = dict({"train": train_acc_history, "val": val_acc_history})
    plot_curve(history, name=os.path.join(ROOT, "models", "loss.png"))
    timer.report()
    print("Best val loss: {:4f}".format(best_loss))


def plot_curve(history: dict, name: str = "loss.png") -> None:
    import matplotlib.pyplot as plt

    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.legend()
    plt.savefig(name)
    plt.close()
