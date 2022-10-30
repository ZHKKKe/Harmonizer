import torch.nn as nn


def add_parser_arguments(parser):
    pass


def task_model():
    return TaskModel


class TaskModel(nn.Module):
    def __init__(self, args=None):
        super(TaskModel, self).__init__()
        self.args = args        
        self.model = None       
        self.param_groups = []  

    def forward(self, inp):
        raise NotImplementedError
