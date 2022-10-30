import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms.functional as tf

from src import model


def load_video_frames(video_path):
    frames = []
    
    vc = cv2.VideoCapture(video_path)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
         rval = False
    if not rval:
        return frames

    frame_num = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    for fdx in range(0, int(frame_num)):
        frames.append(frame)
        rval, frame = vc.read()
    
    return frames


if __name__ == '__main__':
    # define/parse cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--example-path', type=str, required=True, help='')
    parser.add_argument('--pretrained', type=str, default='./pretrained/harmonizer.pth', help='')
    args = parser.parse_known_args()[0]

    print('\n')
    print('================================================================================')
    print('Video Harmonization Demo')
    print('--------------------------------------------------------------------------------')
    print('  - Example Path: {0}'.format(args.example_path))
    print('  - Pretrained Model: {0}'.format(args.pretrained))
    print('================================================================================')
    print('\n')

    # check cmd argments
    if not os.path.exists(args.example_path):
        print('Cannot find the example path: {0}'.format(args.example_path))
        exit()
    if not os.path.exists(args.pretrained):
        print('Cannot find the pretrained model: {0}'.format(args.pretrained))
        exit()

    # create output path
    os.makedirs(os.path.join(args.example_path, 'composite'), exist_ok=True)
    print('The comp videos will be saved in: {0}'.format(os.path.join(args.example_path, 'harmonized')))
    os.makedirs(os.path.join(args.example_path, 'harmonized'), exist_ok=True)
    print('The harmonized videos will be saved in: {0}\n'.format(os.path.join(args.example_path, 'harmonized')))

    # pre-defined arguments
    fps = 25
    ema = 1 - 1 / fps
    cuda = torch.cuda.is_available()

    # create/load the harmonizer model
    print('Create/load Harmonizer...\n')
    harmonizer = model.Harmonizer()
    if cuda:
        harmonizer = harmonizer.cuda()
    harmonizer.load_state_dict(torch.load(args.pretrained), strict=True)
    harmonizer.eval()

    examples = os.listdir(os.path.join(args.example_path, 'foreground'))
    for vdx, example in enumerate(examples):
        print('Process video: {0}...'.format(example))
        
        # define example path
        mask_video_path = os.path.join(args.example_path, 'mask', example)
        fg_video_path = os.path.join(args.example_path, 'foreground', example)
        bg_video_path = os.path.join(args.example_path, 'background', example)
        comp_video_path = os.path.join(args.example_path, 'composite', example)
        harmonized_video_path = os.path.join(args.example_path, 'harmonized', example)

        # read input videos
        mask_frames = load_video_frames(mask_video_path)
        fg_frames = load_video_frames(fg_video_path)
        bg_frames = load_video_frames(bg_video_path)

        # check frame shape
        assert fg_frames[0].shape[:2] == mask_frames[0].shape[:2] == bg_frames[0].shape[:2]
        assert len(mask_frames) == len(fg_frames)

        # define video writer
        h, w = fg_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        comp_vw = cv2.VideoWriter(comp_video_path, fourcc, fps, (w, h))
        harmonized_vw = cv2.VideoWriter(harmonized_video_path, fourcc, fps, (w, h))

        # harmonization
        ema_arguments = None
        pbar = tqdm(range(len(fg_frames)), total=len(fg_frames), unit='frame')
        for i, fdx in enumerate(pbar):
            mask = cv2.cvtColor(mask_frames[fdx % len(mask_frames)], cv2.COLOR_BGR2RGB)
            fg = cv2.cvtColor(fg_frames[fdx % len(fg_frames)], cv2.COLOR_BGR2RGB)
            bg = cv2.cvtColor(bg_frames[fdx % len(bg_frames)], cv2.COLOR_BGR2RGB)

            comp = fg * (mask / 255.0) + bg * (1 - mask / 255.0)

            comp = Image.fromarray(comp.astype(np.uint8))
            mask = Image.fromarray(mask[:, :, 0].astype(np.uint8))

            comp = tf.to_tensor(comp)[None, ...]
            mask = tf.to_tensor(mask)[None, ...]

            if cuda:
                comp = comp.cuda()
                mask = mask.cuda()

            with torch.no_grad():
                arguments = harmonizer.predict_arguments(comp, mask)

                if ema_arguments is None:
                    ema_arguments = list(arguments)
                else:
                    for i, (ema_argument, argument) in enumerate(zip(ema_arguments, arguments)):
                        ema_arguments[i] = ema * ema_argument + (1 - ema) * argument

                harmonized = harmonizer.restore_image(comp, mask, ema_arguments)[-1]

            comp = np.transpose(comp[0].cpu().numpy(), (1, 2, 0)) * 255
            comp = cv2.cvtColor(comp.astype('uint8'), cv2.COLOR_RGB2BGR)
            comp_vw.write(comp)

            harmonized = np.transpose(harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
            harmonized = cv2.cvtColor(harmonized.astype('uint8'), cv2.COLOR_RGB2BGR)
            harmonized_vw.write(harmonized)

        comp_vw.release()
        harmonized_vw.release()

        print('\n')

    print('Finished.')
    print('\n')
