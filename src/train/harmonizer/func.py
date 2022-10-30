import numpy as np
import torch

import skimage

import torchtask

def task_func():
    return HarmonizationFunc


class HarmonizationFunc(torchtask.func_template.TaskFunc):
    def __init__(self, args):
        super(HarmonizationFunc, self).__init__(args)

    def metrics(self, pred_image, gt_image, mask, meters, id_str=''):
        n, c, h, w = pred_image.shape
        
        assert n == 1

        total_pixels = h * w
        fg_pixels = int(torch.sum(mask, dim=(2, 3))[0][0].cpu().numpy())

        pred_image = torch.clamp(pred_image * 255, 0, 255)
        gt_image = torch.clamp(gt_image * 255, 0, 255)

        pred_image = pred_image[0].permute(1, 2, 0).detach().cpu().numpy()
        gt_image = gt_image[0].permute(1, 2, 0).detach().cpu().numpy()
        mask = mask[0].permute(1, 2, 0).detach().cpu().numpy()

        batch_mse = skimage.metrics.mean_squared_error(pred_image, gt_image)
        meters.update('{0}_{1}_mse'.format(id_str, self.METRIC_STR), batch_mse)

        batch_fmse = skimage.metrics.mean_squared_error(pred_image * mask, gt_image * mask) * total_pixels / fg_pixels
        meters.update('{0}_{1}_fmse'.format(id_str, self.METRIC_STR), batch_fmse)

        batch_psnr = skimage.metrics.peak_signal_noise_ratio(pred_image, gt_image, data_range=pred_image.max() - pred_image.min())
        meters.update('{0}_{1}_psnr'.format(id_str, self.METRIC_STR), batch_psnr)
        
        batch_ssim = skimage.metrics.structural_similarity(pred_image, gt_image, multichannel=True)
        meters.update('{0}_{1}_ssim'.format(id_str, self.METRIC_STR), batch_ssim)
