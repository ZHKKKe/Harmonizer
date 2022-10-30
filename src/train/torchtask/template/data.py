import os
import io
from PIL import Image

from torch.utils.data import Dataset


def add_parser_arguments(parser):
    pass


def task_dataset():
    return TaskDataset


class TaskDataset(Dataset):
    def __init__(self, args=None, is_train=True):
        super(TaskDataset, self).__init__()

        self.args = args                
        self.is_train = is_train        
        self.root_dir = None            

        self.sample_list = []           
        self.idxs = []                  

        self.im_loader = ImageLoader()  

        if is_train:
            self.root_dir = list(self.args.trainset.values())[0]
        else:
            self.root_dir = list(self.args.valset.values())[0]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        raise NotImplementedError


class ImageLoader:
    def __init__(self):
        pass

    def load(self, name):
        image = Image.open(name)
        return image
