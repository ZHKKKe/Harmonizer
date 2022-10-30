import re
import argparse

from . import logger


cmdline_strs = None


def parse_args(parser, args_dict):
    global cmdline_strs

    def dict_to_cmdline(key, value):
        if len(key) == 1:
            key = '-{}'.format(key)
        else:
            key = '--{}'.format(re.sub(r'_', '-', key))
        value = str(value)
        return key, value

    cmdline_strs = [dict_to_cmdline(key, value) for key, value in args_dict.items()]
    cmdline_strs = ['{0} = {1}'.format(param[0], param[1]) for param in cmdline_strs]

    cmdline_args = (dict_to_cmdline(key, value) for key, value in args_dict.items())
    cmdline_args = list(sum(cmdline_args, ()))

    return parser.parse_args(cmdline_args)


def print_args():
    global cmdline_strs
    logger.log_info('Experiment args: \n  {0}\n'.format('\n  '.join(cmdline_strs)))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        logger.log_err('str2bool requires a boolean value, but got {0}\n'.format(v))


def str2intlist(v):
    v = v.replace('[', '').replace(']', '').replace('(', '').replace(')', '').split(',')
    int_list = []
    for i in v:
        int_list.append(int(i.strip()))

    return int_list


def str2floatlist(v):
    v = v.replace('[', '').replace(']', '').replace('(', '').replace(')', '').split(',')
    float_list = []
    for f in v:
        float_list.append(float(f.strip()))

    return float_list
