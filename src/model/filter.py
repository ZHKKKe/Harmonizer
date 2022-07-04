import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class BrightnessFilter(nn.Module):
    def __init__(self):
        super(BrightnessFilter, self).__init__()
        self.epsilon = 1e-6

    def forward(self, image, x):
        """
        Arguments:
            image (tensor [n, 3, h, w]): RGB image with pixel values between [0, 1]
            x (tensor [n, 1, 1, 1]): brightness argument with values between [-1, 1]
        """
        
        # convert image from RGB to HSV
        image = kornia.color.rgb_to_hsv(image)
        h = image[:,0:1,:,:]
        s = image[:,1:2,:,:]
        v = image[:,2:3,:,:]
        
        # calculate alpha
        amask = (x >= 0).float()
        alpha = (1 / ((1 - x) + self.epsilon)) * amask + (x + 1) * (1 - amask)

        # adjust the V channel
        v = v * alpha

        # convert image from HSV to RGB
        image = torch.cat((h, s, v), dim=1)
        image = kornia.color.hsv_to_rgb(image)

        # clip pixel values to [0, 1]
        image = torch.clamp(image, 0.0, 1.0)

        return image


class ContrastFilter(nn.Module):
    def __init__(self):
        super(ContrastFilter, self).__init__()

    def forward(self, image, x):
        """
        Arguments:
            image(tensor [n, 3, h, w]): RGB image with pixel values between [0, 1]
            x (tensor [n, 1, 1, 1]): contrast argument with values between [-1, 1]
        """

        # calculate the mean of the image as the threshold
        threshold = torch.mean(image, dim=(1, 2, 3), keepdim=True)

        # pre-process x if it is a positive value
        mask = (x.detach() > 0).float()
        x_ = 255 / (256 - torch.floor(x * 255)) - 1
        x_ = x * (1 - mask) + x_ * mask

        # modify the contrast of the image
        image = image + (image - threshold) * x_

        # clip pixel values to [0, 1]
        image = torch.clamp(image, 0.0, 1.0)

        return image


class SaturationFilter(nn.Module):
    def __init__(self):
        super(SaturationFilter, self).__init__()

        self.epsilon = 1e-6

    def forward(self, image, x):
        """
        Arguments:
            image(tensor [n, 3, h, w]): RGB image with pixel values between [0, 1]
            x (tensor [n, 1, 1, 1]): saturation argument with values between [-1, 1]
        """
        
        # calculate the basic properties of the image
        cmin = torch.min(image, dim=1, keepdim=True)[0]
        cmax = torch.max(image, dim=1, keepdim=True)[0]
        var = cmax - cmin
        ran = cmax + cmin
        mean = ran / 2

        is_positive = (x.detach() >= 0).float()

        # calculate s
        m = (mean < 0.5).float()
        s = (var / (ran + self.epsilon)) * m + (var / (2 - ran + self.epsilon)) * (1 - m)

        # if x is positive
        m = ((x + s) > 1).float()
        a_pos = s * m + (1 - x) * (1 - m)
        a_pos = 1 / (a_pos + self.epsilon) - 1

        # if x is negtive
        a_neg = 1 + x
        
        a = a_pos * is_positive + a_neg * (1 - is_positive)
        image = image * is_positive + mean * (1 - is_positive) + (image - mean) * a

        # clip pixel values to [0, 1]
        image = torch.clamp(image, 0.0, 1.0)

        return image


class TemperatureFilter(nn.Module):
    def __init__(self):
        super(TemperatureFilter, self).__init__()
    
        self.epsilon = 1e-6

    def forward(self, image, x):
        """
        Arguments:
            image(tensor [n, 3, h, w]): RGB image with pixel values between [0, 1]
            x (tensor [n, 1, 1, 1]): color temperature argument with values between [-1, 1]
        """
        # split the R/G/B channels
        R, G, B = image[:, 0:1, ...], image[:, 1:2, ...], image[:, 2:3, ...]

        # calculate the mean of each channel
        meanR = torch.mean(R, dim=(2, 3), keepdim=True)
        meanG = torch.mean(G, dim=(2, 3), keepdim=True)
        meanB = torch.mean(B, dim=(2, 3), keepdim=True)

        # calculate correction factors
        gray = (meanR + meanG + meanB) / 3
        coefR = gray / (meanR + self.epsilon)
        coefG = gray / (meanG + self.epsilon)
        coefB = gray / (meanB + self.epsilon)
        aR = 1 - coefR
        aG = 1 - coefG
        aB = 1 - coefB

        # adjust temperature
        is_positive = (x.detach() > 0).float()
        is_negative = (x.detach() < 0).float()
        is_zero = (x.detach() == 0).float()
        
        meanR_ = meanR + x * torch.sign(x) * is_negative
        meanG_ = meanG + x * torch.sign(x) * 0.5 * (1 - is_zero)
        meanB_ = meanB + x * torch.sign(x) * is_positive
        gray_ = (meanR_ + meanG_ + meanB_) / 3

        coefR_ = gray_ / (meanR_ + self.epsilon) + aR
        coefG_ = gray_ / (meanG_ + self.epsilon) + aG
        coefB_ = gray_ / (meanB_ + self.epsilon) + aB

        R_ = coefR_ * R
        G_ = coefG_ * G
        B_ = coefB_ * B

        # the RGB image with the adjusted brightness
        image = torch.cat((R_, G_, B_), dim=1)

        # clip pixel values to [0, 1]
        image = torch.clamp(image, 0.0, 1.0)

        return image


class HighlightFilter(nn.Module):
    def __init__(self):
        super(HighlightFilter, self).__init__()

    def forward(self, image, x):
        """
        Arguments:
            image(tensor [n, 3, h, w]): RGB image with pixel values between [0, 1]
            x (tensor [n, 1, 1, 1]): highlight argument with values between [-1, 1]
        """

        x = x + 1

        image = kornia.enhance.invert(image, image.detach() * 0 + 1)
        image = torch.clamp(torch.pow(image + 1e-9, x), 0.0, 1.0)
        image = kornia.enhance.invert(image, image.detach() * 0 + 1)

        # clip pixel values to [0, 1]
        image = torch.clamp(image, 0.0, 1.0)

        return image


class ShadowFilter(nn.Module):
    def __init__(self):
        super(ShadowFilter, self).__init__()

    def forward(self, image, x):
        """
        Arguments:
            image(tensor [n, 3, h, w]): RGB image with pixel values between [0, 1]
            x (tensor [n, 1, 1, 1]): shadow argument with values between [-1, 1]
        """

        x = -x + 1
        image = torch.clamp(torch.pow(image + 1e-9, x), 0.0, 1.0)

        # clip pixel values to [0, 1]
        image = torch.clamp(image, 0.0, 1.0)

        return image


class Filter(Enum):
    BRIGHTNESS = 1
    CONTRAST = 2
    SATURATION = 3
    TEMPERATURE = 4
    HIGHLIGHT = 5
    SHADOW = 6


FILTER_MODULES = {
    Filter.BRIGHTNESS: BrightnessFilter,
    Filter.CONTRAST: ContrastFilter,
    Filter.SATURATION: SaturationFilter,
    Filter.TEMPERATURE: TemperatureFilter,
    Filter.HIGHLIGHT: HighlightFilter,
    Filter.SHADOW: ShadowFilter,
}
