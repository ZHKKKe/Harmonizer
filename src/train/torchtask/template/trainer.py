import torch

from torchtask.utils import logger


def add_parser_arguments(parser):
    pass


def task_trainer(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    raise NotImplementedError



class TaskTrainer:
    def __init__(self, args):
        self.args = args                        # arguments required by the task trainer
        self.task_func = None                   # instance of 'TaskFunc' associated with a particular task
        self.meters = logger.AvgMeterSet()      # tool class for logging

        self.models = {}                        # dict of the models required by the task and algorithm
        self.optimizers = {}                    # dict of the optimizers required by the task and algorithm
        self.lrers = {}                         # dict of the learn rate required by the task and algorithm
        self.criterions = {}                    # dict of the criterions required by the task and algorithm

    # ---------------------------------------------------------------------
    # Interface for task proxy
    # ---------------------------------------------------------------------
    
    def build(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        self._build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)

    def train(self, data_loader, epoch):
        self._train(data_loader, epoch)

    def validate(self, data_loader, epoch):
        self._validate(data_loader, epoch)
    
    def save_checkpoint(self, epoch):
        self._save_checkpoint(epoch)
    
    def load_checkpoint(self):
        return self._load_checkpoint()

    # ---------------------------------------------------------------------
    # All task trainer should implement the following functions
    # ---------------------------------------------------------------------

    def _build(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        raise NotImplementedError

    def _train(self, data_loader, epoch):
        raise NotImplementedError

    def _validate(self, data_loader, epoch):
        raise NotImplementedError
        
    def _save_checkpoint(self, epoch):
        raise NotImplementedError
            
    def _load_checkpoint(self):
        raise NotImplementedError
