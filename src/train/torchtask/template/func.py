import torch

from torchtask.utils import logger


def task_func():
    return TaskFunc
    

class TaskFunc:
    # special indetifier for the metric elements
    METRIC_STR = 'metric'
    
    def __init__(self, args=None):
        self.args = args

    # ---------------------------------------------------------------------
    # Functions for All Tasks
    # Following functions are required by all tasks.
    # ---------------------------------------------------------------------

    def metrics(self, pred, gt, inp, meters, id_str=''):
        logger.log_warn('No implementation of the \'metrics\' function for current task.\n'
                        'Please implement it in \'task/xxx/func.py\'.\n')

    def visualize(self, out_path, id_str='', inp=None, pred=None, gt=None):
        logger.log_warn('No implementation of the \'visulize\' function for current task.\n'
                        'Please implement it in \'task/xxx/func.py\'.\n')
         
    # ---------------------------------------------------------------------
