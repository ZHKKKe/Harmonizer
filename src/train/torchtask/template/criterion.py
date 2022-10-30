import torch.nn as nn


def add_parser_arguments(parser):
    pass


def task_criterion():
    return TaskCriterion


class TaskCriterion(nn.Module):
    def __init__(self, args=None):
        super(TaskCriterion, self).__init__()
        
        self.args = args

    def forward(self, pred, gt, inp):
        raise NotImplementedError
