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
    parser.add_argument('--pretrained', type=str, default='./pretrained/enhancer.pth', help='')
    args = parser.parse_known_args()[0]

    print('\n')
    print('================================================================================')
    print('Video Enhancement Demo')
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
    os.makedirs(os.path.join(args.example_path, 'enhanced'), exist_ok=True)
    print('The enhanced videos will be saved in: {0}\n'.format(os.path.join(args.example_path, 'enhanced')))

    # pre-defined arguments
    fps = 25
    ema = 1 - 1 / fps
    cuda = torch.cuda.is_available()

    # create/load the enhancer model
    print('Create/load Enhancer...\n')
    enhancer = model.Enhancer()
    if cuda:
        enhancer = enhancer.cuda()
    enhancer.load_state_dict(torch.load(args.pretrained), strict=True)
    enhancer.eval()

    examples = os.listdir(os.path.join(args.example_path, 'original'))
    for vdx, example in enumerate(examples):
        print('Process video: {0}...'.format(example))
        
        # define example path
        original_video_path = os.path.join(args.example_path, 'original', example)
        enhanced_video_path = os.path.join(args.example_path, 'enhanced', example)

        # read input videos
        original_frames = load_video_frames(original_video_path)

        # define video writer
        h, w = original_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        enhanced_vw = cv2.VideoWriter(enhanced_video_path, fourcc, fps, (w, h))

        # harmonization
        ema_arguments = None
        pbar = tqdm(range(len(original_frames)), total=len(original_frames), unit='frame')
        for i, fdx in enumerate(pbar):
            original = cv2.cvtColor(original_frames[fdx % len(original_frames)], cv2.COLOR_BGR2RGB)
            original = Image.fromarray(original.astype(np.uint8))
            original = tf.to_tensor(original)[None, ...]
            if cuda:
                original = original.cuda()

            # NOTE: all pixels in the mask are equal to 1 as the mask is not used in image enhancement
            mask = original * 0 + 1

            with torch.no_grad():
                arguments = enhancer.predict_arguments(original, mask)

                if ema_arguments is None:
                    ema_arguments = list(arguments)
                else:
                    for i, (ema_argument, argument) in enumerate(zip(ema_arguments, arguments)):
                        ema_arguments[i] = ema * ema_argument + (1 - ema) * argument

                enhanced = enhancer.restore_image(original, mask, ema_arguments)[-1]

            enhanced = np.transpose(enhanced[0].cpu().numpy(), (1, 2, 0)) * 255
            enhanced = cv2.cvtColor(enhanced.astype('uint8'), cv2.COLOR_RGB2BGR)
            enhanced_vw.write(enhanced)

        enhanced_vw.release()

        print('\n')

    print('Finished.')
    print('\n')
