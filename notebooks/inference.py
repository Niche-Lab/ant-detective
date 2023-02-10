# torch imports
import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision import transforms

# native importss
import os
from PIL import Image
from models.model import load_model


# PATH
ROOT = os.path.dirname(os.path.abspath(__file__))
PATH_MODEL = os.path.join(ROOT, "models")
PATH_DATA = os.path.join(ROOT, "data", "peptone_sucrose")
weights = os.path.join(PATH_MODEL, "model.pt")
model = load_model(model="ViT", dir_weights=weights)
model.heads = nn.Identity()

# load tensor
filename = os.path.join(PATH_DATA, "train", "t1_421.JPG")
img = Image.open(filename)
img

transf = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.ColorJitter(hue=0.3, brightness=0.05),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

import matplotlib.pyplot as plt

img_s = transf(img)
plt.imshow(img_s.permute(1, 2, 0))

img_out = model(img_s.unsqueeze(0))
img_out


model._modules["model"]
