
import math

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from torchtask.utils import cmd
from torchtask.nn.func import pytorch_support


""" This file wraps the optimizers used in the script.
"""


VALID_OPTIMIZER = ['sgd', 'rmsprop', 'adam', 'wdadam']


def add_parser_arguments(parser):
    """ Add the arguments related to the optimizer.
    
    This 'add_parser_arguments' function will be called every time.
    Please do not use the argument's name that are already defined in is function.
    The default value '-1' means that the default value corresponding to 
    different LR schedulers will be used.
    """

    parser.add_argument('--lr', type=float, default=-1, metavar='',
                        help='optimizer - learning rate (required by [all])')

    parser.add_argument('--dampening', type=float, default=-1, metavar='',
                        help='optimizer - dampening for momentum (required by [sgd])')
    parser.add_argument('--nesterov', type=cmd.str2bool, default=False, metavar='',
                        help='optimizer - enables Nesterov momentum if True (required by [sgd])')
    parser.add_argument('--weight-decay', type=float, default=-1, metavar='',
                        help='optimizer - weight decay (L2 penalty) (required by [sgd, rmsprop, adam, wdadam])')
    parser.add_argument('--momentum', type=float, default=-1, metavar='',
                        help='optimizer - momentum factor (required by [sgd, rmsprop])')
    parser.add_argument('--alpha', type=float, default=-1, metavar='',
                        help='smoothing constant (required by [rmsprop])')
    parser.add_argument('--centered', type=cmd.str2bool, default=False, metavar='',
                        help='if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance ( required by [rmsprop])')
    parser.add_argument('--eps', type=float, default=-1, metavar='',
                        help='optimizer - term added to the denominator to improve numerical stability  (required by [rmsprop, adam, wdadam])')
    parser.add_argument('--beta1', type=float, default=-1, metavar='',
                        help='optimizer - coefficients used for computing running averages of gradient and its square (required by [adam, wdadam])')
    parser.add_argument('--beta2', type=float, default=-1, metavar='',
                        help='optimizer - coefficients used for computing running averages of gradient and its square (required by [adam, wdadam])')
    parser.add_argument('--amsgrad', type=cmd.str2bool, default=False, metavar='',
                        help='optimizer - use the AMSGrad variant if True (required by [wdadam])')


# ---------------------------------------------------------------------
# Wrapper of Optimizer
# ---------------------------------------------------------------------

def sgd(args):
    """ Wrapper of torch.optim.SGD (PyTorch >= 1.0.0).

    Implements stochastic gradient descent (optionally with momentum).
    """
    args.lr = 0.01 if args.lr == -1 else args.lr
    args.weight_decay = 0 if args.weight_decay == -1 else args.weight_decay
    args.momentum = 0 if args.momentum == -1 else args.momentum
    args.dampening = 0 if args.dampening == -1 else args.dampening
    args.nesterov = False if args.nesterov == False else args.nesterov

    def sgd_wrapper(param_groups):
        pytorch_support(required_version='1.0.0', info_str='Optimizer - SGD')
        return optim.SGD(
            param_groups, 
            lr=args.lr, momentum=args.momentum, dampening=args.dampening,
            weight_decay=args.weight_decay, nesterov=args.nesterov)

    return sgd_wrapper


def rmsprop(args):
    """ Wrapper of torch.optim.RMSprop (PyTorch >= 1.0.0).

    Implements RMSprop algorithm.
    Proposed by G. Hinton in his course.
    The centered version first appears in Generating Sequences With Recurrent Neural Networks.
    """

    args.lr = 0.01 if args.lr == -1 else args.lr
    args.alpha = 0.99 if args.alpha == -1 else args.alpha
    args.eps = 1e-08 if args.eps == -1 else args.eps
    args.weight_decay = 0 if args.weight_decay == -1 else args.weight_decay
    args.momentum = 0 if args.momentum == -1 else args.momentum
    args.centered = False if args.centered == False else args.centered

    def rmsprop_wrapper(param_groups):
        pytorch_support(required_version='1.0.0', info_str='Optimizer - RMSprop')
        return optim.RMSprop(
            param_groups,
            lr=args.lr, alpha=args.alpha, eps=args.eps, weight_decay=args.weight_decay,
            momentum=args.momentum, centered=args.centered)

    return rmsprop_wrapper


def adam(args):
    """ Wrapper of torch.optim.Adam (PyTorch >= 1.0.0).

    Implements Adam algorithm.
    It has been proposed in 'Adam: A Method for Stochastic Optimization'.
    """
    args.lr = 0.001 if args.lr == -1 else args.lr
    args.beta1 = 0.9 if args.beta1 == -1 else args.beta1
    args.beta2 = 0.999 if args.beta2 == -1 else args.beta2
    args.eps = 1e-08 if args.eps == -1 else args.eps
    args.weight_decay = 0.0 if args.weight_decay == -1 else args.weight_decay

    def adam_wrapper(param_groups):
        pytorch_support(required_version='1.0.0', info_str='Optimizer - Adam')
        return optim.Adam(
            param_groups, 
            lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, 
            weight_decay=args.weight_decay)
        
    return adam_wrapper   


def wdadam(args):
    """ Wrapper of torchtask.nn.optimizer.WDAdam (PyTorch >= 1.0.0).

    Implements Adam algorithm with weight decay and AMSGrad.
    """
    args.lr = 0.001 if args.lr == -1 else args.lr
    args.beta1 = 0.9 if args.beta1 == -1 else args.beta1
    args.beta2 = 0.999 if args.beta2 == -1 else args.beta2
    args.eps = 1e-08 if args.eps == -1 else args.eps
    args.weight_decay = 0.0 if args.weight_decay == -1 else args.weight_decay
    args.amsgrad = False if args.amsgrad == False else args.amsgrad

    def wdadam_wrapper(param_groups):
        pytorch_support(required_version='1.0.0', info_str='Optimizer - WDAdam')
        return WDAdam(
            param_groups, 
            lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, 
            weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    return wdadam_wrapper


# ---------------------------------------------------------------------
# Implementation of Optimizer
# ---------------------------------------------------------------------

class WDAdam(Optimizer):
    """ Implements Adam algorithm with weight decay and AMSGrad.
    
    It has been proposed in `Adam: A Method for Stochastic Optimization`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay using the method from
            the paper `Fixing Weight Decay Regularization in Adam` (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {0}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {0}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {0}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {0}".format(betas[1]))
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay / lr, amsgrad=amsgrad)
        super(WDAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(WDAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """ Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')
                amsgrad = group['amsgrad']

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
