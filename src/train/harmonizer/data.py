import os
import cv2
import random
import numpy as np
from PIL import Image

from torchvision import transforms

import torchtask


def add_parser_arguments(parser):
    torchtask.data_template.add_parser_arguments(parser)


def harmonizer_iharmony4():
    return HarmonizerIHarmony4


def original_iharmony4():
    return OriginalIHarmony4


def resize(img, size):
    interp = cv2.INTER_LINEAR

    return Image.fromarray(
        cv2.resize(np.array(img).astype('uint8'), size, interpolation=interp))


im_train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
    transforms.ToTensor(),
])

im_val_transform = transforms.Compose([
    transforms.ToTensor(),
])


class HarmonizerIHarmony4(torchtask.data_template.TaskDataset):
    def __init__(self, args, is_train):
        super(HarmonizerIHarmony4, self).__init__(args, is_train)

        self.im_dir = os.path.join(self.root_dir, 'image')
        self.mask_dir = os.path.join(self.root_dir, 'mask')

        if not os.path.exists(self.mask_dir):
            self.mask_dir = os.path.join(self.root_dir, 'matte')

        self.sample_list = [_ for _ in os.listdir(self.im_dir)]
        self.idxs = [_ for _ in range(0, len(self.sample_list))]

        self.im_size = self.args.im_size

        self.rotation = True if self.is_train else False
        self.fliplr = True if self.is_train else False

    def __getitem__(self, idx):
        image_path = os.path.join(self.im_dir, self.sample_list[idx])
        mask_path = os.path.join(self.mask_dir, self.sample_list[idx])
        
        image = self.im_loader.load(image_path)
        mask = self.im_loader.load(mask_path)

        width, height = image.size

        # resize to self.im_size
        image = resize(image, (self.im_size, self.im_size))
        mask = resize(mask, (self.im_size, self.im_size))

        # convert to np array and scale to [0, 1]
        image = np.array(image).astype('float32') / 255.0
        mask = np.array(mask).astype('float32') / 255.0

        # check image shape
        if len(mask.shape) == 3:
            mask = mask[:, :, -1]

        if len(image.shape) == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]
        
        # random rotate
        rerotation = 0
        if self.rotation and random.randint(0, 1) == 0:
            rotate_num = random.randint(1, 3)
            rerotation = 4 - rotate_num
            image = np.rot90(image, k=rotate_num).copy()
            mask =  np.rot90(mask, k=rotate_num).copy()
        
        # random flip
        if self.fliplr and (random.randint(0, 1) == 0):
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        image = Image.fromarray((image * 255.0).astype('uint8'))
        if self.is_train:
            image = im_train_transform(image)
        else:
            image = im_val_transform(image)

        mask = mask[None, :, :]
        adjusted = image.numpy() * -1

        return (adjusted, mask), (image, )


class OriginalIHarmony4(torchtask.data_template.TaskDataset):
    def __init__(self, args, is_train):
        super(OriginalIHarmony4, self).__init__(args, is_train)

        self.adjusted_dir = os.path.join(self.root_dir, 'comp')
        self.mask_dir = os.path.join(self.root_dir, 'mask')
        self.im_dir = os.path.join(self.root_dir, 'image')

        self.sample_list = [_ for _ in os.listdir(self.adjusted_dir)]
        self.idxs = [_ for _ in range(0, len(self.sample_list))]

        self.im_size = self.args.im_size

        self.rotation = True if self.is_train else False
        self.fliplr = True if self.is_train else False
    
    def __getitem__(self, idx):
        sname = self.sample_list[idx]
        adjusted_path = os.path.join(self.adjusted_dir, sname)
        image_path = os.path.join(self.im_dir, sname)
        mask_path = os.path.join(self.mask_dir, sname)

        if not os.path.exists(image_path):
            prefix = '_'.join(sname.split('_')[:-1])
            image_path = os.path.join(self.im_dir, '{0}.jpg'.format(prefix))
            mask_path = os.path.join(self.mask_dir, '{0}.jpg'.format(prefix))

        adjusted = self.im_loader.load(adjusted_path)
        image = self.im_loader.load(image_path)
        mask = self.im_loader.load(mask_path)

        width, height = image.size


        # resize to self.im_size
        adjusted = resize(adjusted, (self.im_size, self.im_size))
        image = resize(image, (self.im_size, self.im_size))
        mask = resize(mask, (self.im_size, self.im_size))

        # convert to np array and scale to [0, 1]
        adjusted = np.array(adjusted).astype('float32') / 255.0
        image = np.array(image).astype('float32') / 255.0
        mask = np.array(mask).astype('float32') / 255.0

        # check image shape
        if len(mask.shape) == 3:
            mask = mask[:, :, -1]

        if len(image.shape) == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]

        if len(adjusted.shape) == 2:
            adjusted = adjusted[:, :, None]
        if adjusted.shape[2] == 1:
            adjusted = np.repeat(adjusted, 3, axis=2)
        elif adjusted.shape[2] == 4:
            adjusted = adjusted[:, :, 0:3]

        # random rotate
        rerotation = 0
        if self.rotation and random.randint(0, 1) == 0:
            rotate_num = random.randint(1, 3)
            rerotation = 4 - rotate_num
            adjusted = np.rot90(adjusted, k=rotate_num).copy()
            image = np.rot90(image, k=rotate_num).copy()
            mask =  np.rot90(mask, k=rotate_num).copy()
        
        # random flip
        if self.fliplr and (random.randint(0, 1) == 0):
            adjusted = np.fliplr(adjusted).copy()
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        adjusted = Image.fromarray((adjusted * 255.0).astype('uint8'))
        image = Image.fromarray((image * 255.0).astype('uint8'))

        # NOTE: do not add random color adjustement here
        adjusted = im_val_transform(adjusted)
        image = im_val_transform(image)

        mask = mask[None, :, :]

        return (adjusted, mask), (image, )
