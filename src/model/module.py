import cv2
import math
from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F

from .filter import Filter, FILTER_MODULES


class CascadeArgumentRegressor(nn.Module):
    def __init__(self, in_channels, base_channels, out_channels, head_num):
        super(CascadeArgumentRegressor, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.head_num = head_num

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.f = nn.Linear(self.in_channels, 160)
        self.g = nn.Linear(self.in_channels, self.base_channels)

        self.headers = nn.ModuleList()
        for i in range(0, self.head_num):
            self.headers.append(
                nn.ModuleList([
                    nn.Linear(160 + self.base_channels, self.base_channels),
                    nn.Linear(self.base_channels, self.out_channels),
                ])
            )

    def forward(self, x):
        x = self.pool(x)
        n, c, _, _ = x.shape
        x = x.view(n, c)

        f = self.f(x)
        g = self.g(x)

        pred_args = []
        for i in range(0, self.head_num):
            g = self.headers[i][0](torch.cat((f, g), dim=1))
            pred_args.append(self.headers[i][1](g))

        return pred_args


class FilterPerformer(nn.Module):
    def __init__(self, filter_types):
        super(FilterPerformer, self).__init__()

        self.filters = [FILTER_MODULES[filter_type]() for filter_type in filter_types]

    def forward(self):
        pass

    def restore(self, x, mask, arguments):
        assert len(self.filters) == len(arguments)
        
        outputs = []
        _image = x
        for filter, arg in zip(self.filters, arguments):
            _image = filter(_image, arg)
            outputs.append(_image * mask + x * (1 - mask))

        return outputs

    def adjust(self, image, mask, arguments):
        assert len(self.filters) == len(arguments)
        
        outputs = []
        _image = image
        for filter, arg in zip(reversed(self.filters), reversed(arguments)):
            _image = filter(_image, arg)
            outputs.append(_image * mask + image * (1 - mask))

        return outputs
        