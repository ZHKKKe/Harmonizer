import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms.functional as tf

from src import model


if __name__ == '__main__':
    # define/parse cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--example-path', type=str, required=True, help='')
    parser.add_argument('--pretrained', type=str, default='./pretrained/harmonizer.pth', help='')
    args = parser.parse_known_args()[0]

    print('\n')
    print('================================================================================')
    print('Image Harmonization Demo')
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
    os.makedirs(os.path.join(args.example_path, 'harmonized'), exist_ok=True)
    print('The harmonized images will be saved in: {0}\n'.format(os.path.join(args.example_path, 'harmonized')))

    # pre-defined arguments
    cuda = torch.cuda.is_available()
    
    # create/load the harmonizer model
    print('Create/load Harmonizer...')
    harmonizer = model.Harmonizer()
    if cuda:
        harmonizer = harmonizer.cuda()
    harmonizer.load_state_dict(torch.load(args.pretrained), strict=True)
    harmonizer.eval()

    print('Process...')
    examples = os.listdir(os.path.join(args.example_path, 'composite'))
    pbar = tqdm(examples, total=len(examples), unit='example')
    for i, example in enumerate(pbar):
        # load the example
        comp = Image.open(os.path.join(args.example_path, 'composite', example)).convert('RGB')
        mask = Image.open(os.path.join(args.example_path, 'mask', example)).convert('1')
        if comp.size[0] != mask.size[0] or comp.size[1] != mask.size[1]:
            print('The size of the composite image and the mask are inconsistent')
            exit()

        comp = tf.to_tensor(comp)[None, ...]
        mask = tf.to_tensor(mask)[None, ...]

        if cuda:
            comp = comp.cuda()
            mask = mask.cuda()

        # harmonization
        with torch.no_grad():
            arguments = harmonizer.predict_arguments(comp, mask)
            harmonized = harmonizer.restore_image(comp, mask, arguments)[-1]

        # save the result
        harmonized = np.transpose(harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
        harmonized = Image.fromarray(harmonized.astype(np.uint8))
        harmonized.save(os.path.join(args.example_path, 'harmonized', example))

    print('Finished.')
    print('\n')

