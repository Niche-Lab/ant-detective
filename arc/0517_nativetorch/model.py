# native imports
import platform

# torch imports
import torch
import torch.nn as nn

# transformer imports
from transformers import (
    AutoModelForSemanticSegmentation,
    AutoFeatureExtractor,
    DetrForSegmentation,
    DetrFeatureExtractor,
)

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
        print("GPU is available, CUDA used", flush=True)
        str_device = "cuda"
    elif platform.system() == "Darwin":
        print("M1 GPU is available, MPS used", flush=True)
        str_device = "mps"
    else:
        print("GPU is not available, CPU used", flush=True)
        str_device = "cpu"

    device = torch.device(str_device)
    return device


# TRANSFORMER
def init_model(model_name, repo_lbs=None, json_lbs=None) -> nn.Module:
    """
    DESCRIPTION:
    Loading the pretrained models / Deep learning architecture.
    """
    if "mit" in model_name or "segformer" in model_name:
        # nvidia/mit-b0 or nvidia/segformer-b0-finetuned-ade-512-512
        id2label, label2id = get_labels(repo_id=repo_lbs, filename=json_lbs)
        return AutoModelForSemanticSegmentation.from_pretrained(
            model_name, id2label=id2label, label2id=label2id
        )
    elif "detr" in model_name:
        # facebook/detr-resnet-50-panoptic
        return DetrForSegmentation.from_pretrained(model_name)


def get_labels(repo_id, filename):
    """
    # https://huggingface.co/datasets/huggingface/label-files/tree/main
    repo_id = "huggingface/label-files"
    filename = "ade20k-id2label.json" # was "ade20k-hf-doc-builder.json"
    """
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset")))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def get_features_ext(model_name):
    """
    DESCRIPTION:
    This function returns the feature of the pretrained model. ##
    """

    if "mit" in model_name or "segformer" in model_name:
        # nvidia/mit-b0 or nvidia/segformer-b0-finetuned-ade-512-512
        return AutoFeatureExtractor.from_pretrained(model_name, return_tensors=True)
    elif "detr" in model_name:
        # facebook/detr-resnet-50-panoptic
        return DetrFeatureExtractor.from_pretrained(model_name)
