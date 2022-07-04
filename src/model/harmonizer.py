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

        self.backbone = EfficientBackbone.from_name('efficientnet-b0')
        self.regressor = CascadeArgumentRegressor(1280, 160, 1, len(self.filter_types))
        self.performer = FilterPerformer(self.filter_types)

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
        return self.performer.restore(comp, mask, arguments)[-1]
