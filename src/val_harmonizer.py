import os
import skimage
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf

from . import model


def load_iHarmony4_subset(dataset_dir, mode):
    if not mode in ['train', 'test']:
        print('Invalid mode: {0} for the dataset: {1}'.format(mode, dataset_dir))
        exit()

    sample_names = []
    with open(os.path.join(dataset_dir, '{0}_{1}.txt'.format(dataset_dir.split('/')[-1], mode)), 'r') as f:
        sample_names = [_.strip() for _ in f.readlines()]

    comp_dir = os.path.join(dataset_dir, 'composite_images')
    mask_dir = os.path.join(dataset_dir, 'masks')
    real_dir = os.path.join(dataset_dir, 'real_images')

    samples = []
    comp_names = os.listdir(comp_dir)
    for comp_name in comp_names:
        if comp_name in sample_names:
            mask_name = '_'.join(comp_name.split('_')[:-1]) + '.png'
            real_name = '_'.join(comp_name.split('_')[:-2]) + '.jpg'

            sample = {
                'comp': os.path.join(comp_dir, comp_name),
                'mask': os.path.join(mask_dir, mask_name),
                'real': os.path.join(real_dir, real_name),
            }

            samples.append(sample)

    return samples


def calc_metrics(pred, gt, mask):
    n, c, h, w = pred.shape
    assert n == 1
    total_pixels = h * w
    fg_pixels = int(torch.sum(mask, dim=(2, 3))[0][0].cpu().numpy())

    pred = torch.clamp(pred * 255, 0, 255)
    gt = torch.clamp(gt * 255, 0, 255)

    pred = pred[0].permute(1, 2, 0).cpu().numpy()
    gt = gt[0].permute(1, 2, 0).cpu().numpy()
    mask = mask[0].permute(1, 2, 0).cpu().numpy()

    mse = skimage.metrics.mean_squared_error(pred, gt)
    fmse = skimage.metrics.mean_squared_error(pred * mask, gt * mask) * total_pixels / fg_pixels
    psnr = skimage.metrics.peak_signal_noise_ratio(pred, gt, data_range=pred.max() - pred.min())
    ssim = skimage.metrics.structural_similarity(pred, gt, multichannel=True)

    return mse, fmse, psnr, ssim


if __name__ == '__main__':
    # check dataset dir
    DATASET_DIR = './dataset'
    if not os.path.exists(DATASET_DIR):
        print('Cannot find the dataset dir')
        exit()

    # supported image harmonization validation datasets
    DATASETS = {
        'HCOCO': os.path.join(DATASET_DIR, 'harmonization/iHarmony4/HCOCO'),
        'HFlickr': os.path.join(DATASET_DIR, 'harmonization/iHarmony4/HFlickr'),
        'HAdobe5k': os.path.join(DATASET_DIR, 'harmonization/iHarmony4/HAdobe5k'),
        'Hday2night': os.path.join(DATASET_DIR, 'harmonization/iHarmony4/Hday2night'),
    }

    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default='./pretrained/harmonizer.pth', help='')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, choices=DATASETS.keys(), help='')
    parser.add_argument('--metric-size', type=int, default=0, help='')
    args = parser.parse_known_args()[0]

    # pre-process the required arguments
    metric_size = (args.metric_size, args.metric_size) if args.metric_size > 0 else None
    cuda = torch.cuda.is_available()

    # print arguments
    print('\n')
    print('Evaluation Harmonizer:')
    print('  - Pretrained Model: {0}'.format(args.pretrained))
    print('  - Validation Datasets: {0}'.format(args.datasets))
    print('  - Metric Calculation Size: {0}'.format(metric_size if args.metric_size > 0 else 'original'))

    # create/load the harmonizer model
    harmonizer = model.Harmonizer()
    if cuda:
        harmonizer = harmonizer.cuda()
    harmonizer.load_state_dict(torch.load(args.pretrained), strict=True)
    harmonizer.eval()

    # load validation datasets
    datasets = {}
    for d in args.datasets:
        datasets[d] = load_iHarmony4_subset(DATASETS[d], 'test')

    # validation
    metrics = {}
    for dkey, dvalue in datasets.items():
        print('\n')
        print('================================================================================')
        print('Validation Dataset: {0}'.format(dkey))
        print('--------------------------------------------------------------------------------')
        metric = {'MSE': 0, 'fMSE': 0, 'PSNR': 0, 'SSIM': 0}
        sample_num = len(dvalue)
        pbar = tqdm(dvalue, total=sample_num, unit='sample')

        for i, sample in enumerate(pbar):
            # load inputs
            comp = Image.open(sample['comp']).convert('RGB')
            mask = Image.open(sample['mask']).convert('1')
            image = Image.open(sample['real']).convert('RGB')

            # prepare inputs for argument prediction
            _comp = tf.to_tensor(comp)[None, ...]
            _mask = tf.to_tensor(mask)[None, ...]
            _image = tf.to_tensor(image)[None, ...]
            if cuda:
                _comp, _mask, _image = _comp.cuda(), _mask.cuda(), _image.cuda()

            # predict arguments
            with torch.no_grad():
                arguments = harmonizer.predict_arguments(_comp, _mask)

            # prepare inputs for metric calculation
            if metric_size is not None:
                _comp = tf.to_tensor(tf.resize(comp, metric_size))[None, ...]
                _mask = tf.to_tensor(tf.resize(mask, metric_size))[None, ...]
                _image = tf.to_tensor(tf.resize(image, metric_size))[None, ...]
                if cuda:
                    _comp, _mask, _image = _comp.cuda(), _mask.cuda(), _image.cuda()

            with torch.no_grad():
                _harmonized = harmonizer.restore_image(_comp, _mask, arguments)

            # calculate metrics
            mse, fmse, psnr, ssim = calc_metrics(_harmonized, _image, _mask)
            
            metric['MSE'] += mse
            metric['fMSE'] += fmse
            metric['PSNR'] += psnr
            metric['SSIM'] += ssim
            pbar.set_description('MSE: {0:.4f}   fMSE: {1:.4f}   PSNR: {2:.4f}   SSIM: {3:.4f}'.format(
                metric['MSE']/(i+1), metric['fMSE']/(i+1), metric['PSNR']/(i+1), metric['SSIM']/(i+1)))
        
        print('--------------------------------------------------------------------------------')
        print('{0} - MSE: {1:.4f}   fMSE: {2:.4f}   PSNR: {3:.4f}   SSIM: {4:.4f}'.format(
            dkey, metric['MSE']/sample_num, metric['fMSE']/sample_num, metric['PSNR']/sample_num, metric['SSIM']/sample_num))
        print('================================================================================')

        metrics[dkey] = metric

    sample_num = sum([len(dvalue) for dvalue in datasets.values()])
    mse = sum([metric['MSE'] for metric in metrics.values()]) / sample_num
    fmse = sum([metric['fMSE'] for metric in metrics.values()]) / sample_num
    psnr = sum([metric['PSNR'] for metric in metrics.values()]) / sample_num
    ssim = sum([metric['SSIM'] for metric in metrics.values()]) / sample_num

    print('\n')
    print('================================================================================')
    print('All - MSE: {0:.4f}   fMSE: {1:.4f}   PSNR: {2:.4f}   SSIM: {3:.4f}'.format(mse, fmse, psnr, ssim))
    print('================================================================================')
    print('\n')
