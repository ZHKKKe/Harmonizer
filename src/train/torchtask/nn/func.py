import numpy as np

import torch

from torchtask.utils import logger


""" This file provides tool functions for deep learning.
"""


def sigmoid_rampup(current, rampup_length):
    """ Exponential rampup from https://arxiv.org/abs/1610.02242 . 
    """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))



def split_tensor_tuple(ttuple, start, end, reduce_dim=False):
    """ Slice each tensor in the input tuple by channel-dim.

    Arguments:
        ttuple (tuple): tuple of a torch.Tensor
        start (int): start index of slicing
        end (int): end index of slicing
        reduce_dim (bool): whether reduce the channel-dim when end - start == 1
    
    Returns:
        tuple: a sliced tensor tuple
    """

    result = []

    if reduce_dim:
        assert end - start == 1

    for t in ttuple:
        if end - start == 1 and reduce_dim:
            result.append(t[start, ...])       
        else:
            result.append(t[start:end, ...])

    return tuple(result)


def combine_tensor_tuple(ttuple1, ttuple2, dim):
    result = []

    assert len(ttuple1) == len(ttuple2)

    for t1, t2 in zip(ttuple1, ttuple2):
        result.append(torch.cat((t1, t2), dim=dim))

    return tuple(result)


def create_model(mclass, mname, **kwargs):
    """ Create a nn.Module and setup it on multiple GPUs.
    """
    model = mclass(**kwargs)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    
    logger.log_info('  ' + '=' * 76 + '\n  {0} parameters \n{1}'.format(mname, model_str(model)))
    return model


def model_str(module):
    """ Output model structure and parameters number as strings.
    """
    row_format = '  {name:<40} {shape:>20} = {total_size:>12,d}'
    lines = ['  ' + '-' * 76,]

    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(name=name,
            shape=' * '.join(str(p) for p in param.size()), total_size=param.numel()))

    lines.append('  ' + '-' * 76)
    lines.append(row_format.format(name='all parameters', shape='sum of above',
        total_size=sum(int(param.numel()) for name, param in params)))
    lines.append('  ' + '=' * 76)
    lines.append('')

    return '\n'.join(lines)


def pytorch_support(required_version='1.0.0', info_str=''):
    if torch.__version__ < required_version:
        logger.log_err('{0} required PyTorch >= {1}\n'
                       'However, current PyTorch == {2}\n'
                       .format(info_str, required_version, torch.__version__))
    else:
        return True