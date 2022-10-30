"""
This EfficientNet implementation comes from:
    Author: lukemelas (github username)
    Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
"""

import torch
import torch.nn as nn

from .model import EfficientNet
from .utils import round_filters, get_same_padding_conv2d


# for EfficientNet
class EfficientBackbone(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super(EfficientBackbone, self).__init__(blocks_args, global_params)

        self.enc_channels = [16, 24, 40, 112, 1280]

        # ------------------------------------------------------------
        # delete the useless layers
        # ------------------------------------------------------------
        del self._conv_stem
        del self._bn0
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # parameters for the input layers
        # ------------------------------------------------------------
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        in_channels = 4
        out_channels = round_filters(32, self._global_params)
        out_channels = int(out_channels / 2)
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # define the input layers
        # ------------------------------------------------------------
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_fg = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn_fg = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._conv_bg = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn_bg = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        # ------------------------------------------------------------

    def forward(self, xfg, xbg):
        xfg = self._swish(self._bn_fg(self._conv_fg(xfg)))
        xbg = self._swish(self._bn_bg(self._conv_bg(xbg)))

        x = torch.cat((xfg, xbg), dim=1)

        block_outputs = []
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            block_outputs.append(x)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        
        return block_outputs[0], block_outputs[2], block_outputs[4], block_outputs[10], x


# for EfficientNet
class EfficientBackboneCommon(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super(EfficientBackboneCommon, self).__init__(blocks_args, global_params)

        self.enc_channels = [16, 24, 40, 112, 1280]

        # ------------------------------------------------------------
        # delete the useless layers
        # ------------------------------------------------------------
        del self._conv_stem
        del self._bn0
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # parameters for the input layers
        # ------------------------------------------------------------
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # define the input layers
        # ------------------------------------------------------------
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        # ------------------------------------------------------------

    def forward(self, x):
        x = self._swish(self._bn(self._conv(x)))

        block_outputs = []
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            block_outputs.append(x)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        
        return block_outputs[0], block_outputs[2], block_outputs[4], block_outputs[10], x
