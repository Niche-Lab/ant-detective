import torch
import torch.nn as nn


class Niche_Model(nn.Module):
    def __init__(self):
        super(Niche_Model, self).__init__()

    def check_param(self):
        print("Params to learn:")
        params_to_update = []
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
