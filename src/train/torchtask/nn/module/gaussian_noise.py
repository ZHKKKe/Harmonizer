import random

import torch
import torch.nn as nn


class GaussianNoiseLayer(nn.Module):
    """ Add Gaussian noise to a 4D tensor
    """

    def __init__(self, std):
        super(GaussianNoiseLayer, self).__init__()
        self.std = std
        self.noise = torch.zeros(0)
        self.enable = False if self.std is None else True

    def forward(self, inp):
        if not self.enable:
            return inp

        if self.noise.shape != inp.shape:
            self.noise = torch.zeros(inp.shape).cuda()
        self.noise.data.normal_(0, std=random.uniform(0, self.std))

        imax = inp.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        imin = inp.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        
        # normalize to [0, 1]
        inp.sub_(imin).div_(imax - imin + 1e-9)
        # add noise
        inp.add_(self.noise)
        # clip to [0, 1]
        upper_bound = (inp > 1.0).float()
        lower_bound = (inp < 0.0).float()
        inp.mul_(1 - upper_bound).add_(upper_bound)
        inp.mul_(1 - lower_bound)
        # de-normalize
        inp.mul_(imax - imin + 1e-9).add_(imin)

        return inp
