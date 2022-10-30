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
    parser.add_argument('--pretrained', type=str, default='./pretrained/enhancer.pth', help='')
    args = parser.parse_known_args()[0]

    print('\n')
    print('================================================================================')
    print('Image Enhancement Demo')
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
    print('The enhanced images will be saved in: {0}\n'.format(os.path.join(args.example_path, 'enhanced')))

    # pre-defined arguments
    cuda = torch.cuda.is_available()
    
    # create/load the Enhancer model 
    print('Create/load Enhancer...')
    enhancer = model.Enhancer()
    if cuda:
        enhancer = enhancer.cuda()
    enhancer.load_state_dict(torch.load(args.pretrained), strict=True)
    enhancer.eval()

    print('Process...')
    examples = os.listdir(os.path.join(args.example_path, 'original'))
    pbar = tqdm(examples, total=len(examples), unit='example')
    for i, example in enumerate(pbar):
        # load the example
        original = Image.open(os.path.join(args.example_path, 'original', example)).convert('RGB')
        original = tf.to_tensor(original)[None, ...]
        
        # NOTE: all pixels in the mask are equal to 1 as the mask is not used in image enhancement
        mask = original * 0 + 1

        if cuda:
            original = original.cuda()
            mask = mask.cuda()
        
        # enhancement
        with torch.no_grad():
            arguments = enhancer.predict_arguments(original, mask)
            enhanced = enhancer.restore_image(original, mask, arguments)[-1]

        # save the result
        enhanced = np.transpose(enhanced[0].cpu().numpy(), (1, 2, 0)) * 255
        enhanced = Image.fromarray(enhanced.astype(np.uint8))
        enhanced.save(os.path.join(args.example_path, 'enhanced', example))
    
    print('Finished.')
    print('\n')