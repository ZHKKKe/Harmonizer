import math

import torch
import torch.optim as optim

from torchtask.utils import cmd, logger
from torchtask.nn.func import pytorch_support


""" This file wraps the learning rate schedulers used in the script.
"""


EPOCH_LRERS = ['steplr', 'multisteplr', 'exponentiallr', 'cosineannealinglr']
ITER_LRERS = ['polynomiallr']
VALID_LRER = EPOCH_LRERS + ITER_LRERS


def add_parser_arguments(parser):
    """ Add the arguments related to the learning rate (LR) schedulers.
    
    This 'add_parser_arguments' function will be called every time.
    Please do not use the argument's name that are already defined in is function.
    The default value '-1' means that the default value corresponding to 
    different LR schedulers will be used.
    """

    parser.add_argument('--last-epoch', type=int, default=-1, metavar='',
                        help='lr scheduler - the index of last epoch required by [all]')
    
    parser.add_argument('--step-size', type=int, default=-1, metavar='',
                        help='lr scheduler - period (epoch) of learning rate decay required by [steplr]')
    parser.add_argument('--milestones', type=cmd.str2intlist, default=[], metavar='',
                        help='lr scheduler - increased list of epoch indices required by [multisteplr]')
    parser.add_argument('--gamma', type=float, default=-1, metavar='',
                        help='lr scheduler - multiplicative factor of learning rate decay required by [steplr, multisteplr, exponentiallr]')

    parser.add_argument('--T-max', type=int, default=-1, metavar='',
                        help='lr scheduler - maximum number of epochs required by [cosineannealinglr]')
    parser.add_argument('--eta-min', type=float, default=-1, metavar='',
                        help='lr scheduler - minimum learning rate required by [cosineannealinglr]')

    parser.add_argument('--power', type=float, default=-1, metavar='',
                        help='lr scheduler - power factor of learning rate decay required by [polynomiallr]')


# ---------------------------------------------------------------------
# Wrapper of Learning Rate Scheduler
# ---------------------------------------------------------------------

def steplr(args):
    """ Wrapper of torch.optim.lr_scheduler.StepLR (PyTorch >= 1.0.0).

    Sets the learning rate of each parameter group to the initial lr decayed by gamma every 
    step_size epochs. When last_epoch=-1, sets initial lr as lr.
    """
    args.step_size = args.epochs if args.step_size == -1 else args.step_size
    args.gamma = 0.1 if args.gamma == -1 else args.gamma
    args.last_epoch = -1 if args.last_epoch == -1 else args.last_epoch

    def steplr_wrapper(optimizer):
        pytorch_support(required_version='1.0.0', info_str='LRScheduler - StepLR')
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=args.last_epoch)
    
    return steplr_wrapper


def multisteplr(args):
    """ Wrapper of torch.optim.lr_scheduler.MultiStepLR (PyTorch >= 1.0.0).

    Set the learning rate of each parameter group to the initial lr decayed by gamma once the 
    number of epoch reaches one of the milestones. When last_epoch=-1, sets initial lr as lr.
    """
    args.milestones = [i for i in range(1, args.epochs)] if args.milestones == [] else args.milestones
    args.gamma = 0.1 if args.gamma == -1 else args.gamma
    args.last_epoch = -1 if args.last_epoch == -1 else args.last_epoch
    
    def multisteplr_wrapper(optimizer):
        pytorch_support(required_version='1.0.0', info_str='LRScheduler - MultiStepLR')
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma, last_epoch=args.last_epoch)

    return multisteplr_wrapper


def exponentiallr(args):
    """ Wrapper of torch.optim.lr_scheduler.ExponentialLR (PyTorch >= 1.0.0).

    Set the learning rate of each parameter group to the initial lr decayed by gamma every epoch. 
    When last_epoch=-1, sets initial lr as lr.
    """
    args.gamma = 0.1 if args.gamma == -1 else args.gamma
    args.last_epoch = -1 if args.last_epoch == -1 else args.last_epoch

    def exponentiallr_wrapper(optimizer):
        pytorch_support(required_version='1.0.0', info_str='LRScheduler - ExponentialLR')
        return optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.gamma, last_epoch=args.last_epoch)
    
    return exponentiallr_wrapper


def cosineannealinglr(args):
    """ Wrapper of torch.optim.lr_schduler.CosineAnnealingLR (PyTorch >= 1.0.0).

    Set the learning rate of each parameter group using a cosine annealing schedule.
    When last_epoch=-1, sets initial lr as lr.
    """
    args.T_max = args.epochs if args.T_max == -1 else args.T_max
    args.eta_min = 0 if args.eta_min == -1 else args.eta_min
    args.last_epoch = -1 if args.last_epoch == -1 else args.last_epoch

    def cosineannealinglr_wrapper(optimizer):
        pytorch_support(required_version='1.0.0', info_str='LRScheduler - CosineAnnealingLR')
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.T_max, eta_min=args.eta_min, last_epoch=args.last_epoch)

    return cosineannealinglr_wrapper


def polynomiallr(args):
    """ Wrapper of torchtask.nn.lrer.PolynomialLR (PyTorch >= 1.0.0).

    Set the learning rate of each parmeter group to the initial lr decayed by power every 
    iteration. When last_epoch=-1, sets initial lr as lr.
    """
    args.power = 0.9 if args.power == -1 else args.power
    args.last_epoch = -1 if args.last_epoch == -1 else args.last_epoch

    def polynomiallr_wrapper(optimizer):
        pytorch_support(required_version='1.0.0', info_str='LRScheduler - PolynomialLR')
        return PolynomialLR(optimizer, epochs=args.epochs, iters_per_epoch=args.iters_per_epoch, 
                            power=args.power, last_epoch=args.last_epoch)
    
    return polynomiallr_wrapper


# ---------------------------------------------------------------------
# Implementation of Learning Rate Scheduler
# ---------------------------------------------------------------------

class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """ Polynomial decay learning rate scheduler.
    """

    def __init__(self, optimizer, epochs, iters_per_epoch, power=0.9, last_epoch=-1):
        self.epochs = epochs
        self.iters_per_epoch = iters_per_epoch
        self.max_iters = self.epochs * self.iters_per_epoch
        self.cur_iter = 0
        self.power = power
        self.is_warn = False
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * ((1 - float(self.cur_iter) / self.max_iters) ** self.power) 
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is not None and epoch != 0:
            # update lr after each epoch if epoch is given
            # after each epoch, set epoch += 1 and call this function 
            if not self.is_warn:
                logger.log_warn('PolynomialLR is designed for updating learning rate after each iteration.\n'
                                'However, it will be updated after each epoch now, please be careful.\n')
                self.is_warn = True

            self.last_epoch = epoch
            assert self.last_epoch <= self.epochs
            self.cur_iter = self.last_epoch * self.iters_per_epoch

        elif epoch is None:
            # update lr after each iteration if epoch is None
            self.cur_iter += 1
            self.last_epoch = math.floor(self.cur_iter / self.iters_per_epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
