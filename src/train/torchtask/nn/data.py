import os
import itertools
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


""" This file implements dataset wrappers and batch samplers for TorchTask.
"""


class _TorchTaskDatasetWrapper(Dataset):
    """ This is the superclass of TorchTask dataset wrapper.
    """

    def __init__(self):
        super(_TorchTaskDatasetWrapper, self).__init__()

        self.labeled_idxs = []      # index of the labeled data
        self.additional_idxs = []    # index of the additional data


class SplitUnlabeledWrapper(_TorchTaskDatasetWrapper):
    """ Split the fully labeled dataset into a labeled subset and an 
        additional dataset based on a given sublabeled prefix list. 
    
    For a fully labeled dataset, a common operation is to remove the labels 
    of some samples and treat them as the additional samples. 

    This dataset wrapper implements the dataset-split operation by using 
    the given sublabeled prefix list. Samples whose prefix in the list 
    are treated as the labeled samples, while others samples are treated as 
    the additional samples.
    """

    def __init__(self, dataset, sublabeled_prefix, ignore_additional=False):
        super(SplitUnlabeledWrapper, self).__init__()

        self.dataset = dataset
        self.sublabeled_prefix = sublabeled_prefix
        self.ignore_additional = ignore_additional

        self._split_labeled()

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _split_labeled(self):
        labeled_list, additional_list = [], []
        for img in self.dataset.sample_list:
            is_labeled = False
            for pdx, prefix in enumerate(self.sublabeled_prefix):
                if img.startswith(prefix):
                    labeled_list.append(img)
                    is_labeled = True
                    break

            if not is_labeled:
                additional_list.append(img)

        labeled_size, additional_size = len(labeled_list), len(additional_list)
        assert labeled_size + additional_size == len(self.dataset.sample_list)          
        
        if self.ignore_additional:
            self.dataset.sample_list = labeled_list
            self.dataset.idxs = [_ for _ in range(0, len(self.dataset.sample_list))]
            self.labeled_idxs = self.dataset.idxs
            self.additional_idxs = []
        else:
            self.dataset.sample_list = labeled_list + additional_list
            self.dataset.idxs = [_ for _ in range(0, len(self.dataset.sample_list))]
            self.labeled_idxs = [_ for _ in range(0, labeled_size)]
            self.additional_idxs = [_ + labeled_size for _ in range(0, additional_size)]


class JointDatasetsWrapper(_TorchTaskDatasetWrapper):
    """ Combine several datasets (can be labeled or additional) into one dataset.
    
    This dataset wrapper will combine multiple given dataset into one big dataset.
    The new dataset consists of a labeled subset and an additional subset.
    """

    def __init__(self, labeled_datasets, additional_datasets, ignore_additional=False):
        super(JointDatasetsWrapper, self).__init__()

        self.labeled_datasets = labeled_datasets
        self.additional_datasets = additional_datasets
        self.ignore_additional = ignore_additional

        self.labeled_datasets_size = [len(d) for d in self.labeled_datasets]
        self.additional_datasets_size = [len(d) for d in self.additional_datasets]

        self.labeled_size = np.sum(np.asarray(self.labeled_datasets_size))        
        self.labeled_idxs = [_ for _ in range(0, self.labeled_size)]
        
        self.additional_size = 0
        if not self.ignore_additional:
            self.additional_size = np.sum(np.asarray(self.additional_datasets_size))
            self.additional_idxs = [self.labeled_size + _ for _ in range(0, self.additional_size)]

    def __len__(self):
        return int(self.labeled_size + self.additional_size)

    def __getitem__(self, idx):
        assert 0 <= idx < self.__len__()

        if idx >= self.labeled_size:
            idx -= self.labeled_size
            datasets = self.additional_datasets
            datasets_size = self.additional_datasets_size
        else:
            datasets = self.labeled_datasets
            datasets_size = self.labeled_datasets_size

        accumulated_idxs = 0
        for ddx, dsize in enumerate(datasets_size):
            accumulated_idxs += dsize
            if idx < accumulated_idxs:
                return datasets[ddx].__getitem__(idx - (accumulated_idxs - dsize))


class TwoStreamBatchSampler(Sampler):
    """ This two stream batch sampler is used to read data from '_TorchTaskDatasetWrapper'.

    It iterates two sets of indices simultaneously to read mini-batch for TorchTask.
    There are two sets of indices: 
        labeled_idxs, additional_idxs
    An 'epoch' is defined by going through the longer indices once.
    In each 'epoch', the shorter indices are iterated through as many times as needed.
    """

    def __init__(self, labeled_idxs, additional_idxs, labeled_batch_size, additional_batch_size, short_ep=False):
        self.labeled_idxs = labeled_idxs
        self.additional_idxs = additional_idxs
        self.labeled_batch_size = labeled_batch_size
        self.additional_batch_size = additional_batch_size

        assert len(self.labeled_idxs) >= self.labeled_batch_size > 0
        assert len(self.additional_idxs) >= self.additional_batch_size > 0

        self.additional_batchs = len(self.additional_idxs) // self.additional_batch_size
        self.labeled_batchs = len(self.labeled_idxs) // self.labeled_batch_size

        self.short_ep = short_ep

    def __iter__(self):
        if not self.short_ep:
            if self.additional_batchs >= self.labeled_batchs:
                additional_iter = self.iterate_once(self.additional_idxs)
                labeled_iter = self.iterate_eternally(self.labeled_idxs)
            else:
                additional_iter = self.iterate_eternally(self.additional_idxs)
                labeled_iter = self.iterate_once(self.labeled_idxs)
        else:
            if self.additional_batchs >= self.labeled_batchs:
                additional_iter = self.iterate_eternally(self.additional_idxs)
                labeled_iter = self.iterate_once(self.labeled_idxs)
            else:
                additional_iter = self.iterate_once(self.additional_idxs)
                labeled_iter = self.iterate_eternally(self.labeled_idxs)

        return (labeled_batch + additional_batch
                for (labeled_batch, additional_batch) in zip(
                    self.grouper(labeled_iter, self.labeled_batch_size),
                    self.grouper(additional_iter, self.additional_batch_size)))

    def __len__(self):
        if self.short_ep:
            return min(self.additional_batchs, self.labeled_batchs)
        else:
            return max(self.additional_batchs, self.labeled_batchs)

    def iterate_once(self, iterable):
        return np.random.permutation(iterable)

    def iterate_eternally(self, indices):
        def infinite_shuffles():
            while True:
                yield np.random.permutation(indices)

        return itertools.chain.from_iterable(infinite_shuffles())

    def grouper(self, iterable, n):
        # e.g., grouper('ABCDEFG', 3) --> ABC DEF"
        args = [iter(iterable)] * n
        return zip(*args)
