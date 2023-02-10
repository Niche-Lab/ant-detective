# native imports
import platform

# torch imports
import torch
import torch.nn as nn

# local modules
from .ViT import Niche_ViT


def freeze_param(model: nn.Module, freeze: bool = True) -> None:
    """
    freeze = True -> freeze all parameters
    """
    for param in model.parameters():
        param.requires_grad = not freeze


def load_model(model: str, dir_weights: str) -> nn.Module:
    # select model
    if model == "ViT":
        model = Niche_ViT()

    # load existing weights
    load_weights(model, dir_weights)
    return model


def load_weights(model: nn.Module, dir_weights: str) -> nn.Module:
    if dir_weights:
        try:
            model.load_state_dict(torch.load(dir_weights))
            print("Weights loaded", flush=True)
        except Exception as e:
            print(e, flush=True)
            print(
                "Weights not found or not compatible, using original weights",
                flush=True,
            )
    return model


def save_model(model: nn.Module, dir_weights: str) -> None:
    torch.save(model.state_dict(), dir_weights)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("GPU is available", flush=True)
        str_device = "cuda"
        torch.cuda.empty_cache()
    elif platform.system() == "Darwin":
        print("GPU is not available, MPS used", flush=True)
        str_device = "mps"
    else:
        print("GPU is not available, CPU used", flush=True)
        str_device = "cpu"

    device = torch.device(str_device)
    return device
