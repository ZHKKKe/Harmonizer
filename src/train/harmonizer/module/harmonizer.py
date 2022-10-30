import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf

from .filter import Filter
from .backbone import EfficientBackbone
from .module import CascadeArgumentRegressor, FilterPerformer


class Harmonizer(nn.Module):
    def __init__(self):
        super(Harmonizer, self).__init__()
        
        self.input_size = (256, 256)
        self.filter_types = [
            Filter.TEMPERATURE,
            Filter.BRIGHTNESS,
            Filter.CONTRAST,
            Filter.SATURATION,
            Filter.HIGHLIGHT,
            Filter.SHADOW,
        ]
        self.filter_argument_ranges = [
            0.3,
            0.5,
            0.5,
            0.6,
            0.4,
            0.4,
        ]

        self.backbone = EfficientBackbone.from_name('efficientnet-b0')
        self.regressor = CascadeArgumentRegressor(1280, 160, 1, len(self.filter_types))
        self.performer = FilterPerformer(self.filter_types)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        self.backbone = EfficientBackbone.from_pretrained('efficientnet-b0')

    def forward(self, comp, mask):
        arguments = self.predict_arguments(comp, mask)
        pred = self.restore_image(comp, mask, arguments)
        return pred

    def predict_arguments(self, comp, mask):
        comp = F.interpolate(comp, self.input_size, mode='bilinear', align_corners=False)
        mask = F.interpolate(mask, self.input_size, mode='bilinear', align_corners=False)

        fg = torch.cat((comp, mask), dim=1)
        bg = torch.cat((comp, (1 - mask)), dim=1)

        enc2x, enc4x, enc8x, enc16x, enc32x = self.backbone(fg, bg)
        arguments = self.regressor(enc32x)
        return arguments

    def restore_image(self, comp, mask, arguments):
        assert len(arguments) == len(self.filter_types)
        
        arguments = [torch.clamp(arg, -1, 1).view(-1, 1, 1, 1) for arg in arguments]
        return self.performer.restore(comp, mask, arguments)

    def adjust_image(self, image, mask, arguments):
        assert len(arguments) == len(self.filter_types)

        arguments = [(torch.clamp(arg, -1, 1) * r).view(-1, 1, 1, 1) \
            for arg, r in zip(arguments, self.filter_argument_ranges)]
        return self.performer.adjust(image, mask, arguments)

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, bn):
        if bn.weight is not None:
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
