import sys
import argparse


from torchtask.utils import cmd
from torchtask.nn import optimizer, lrer
from torchtask.nn.func import pytorch_support


def create_parser():
    parser = argparse.ArgumentParser(description='TorchTask Script Parser')

    optimizer.add_parser_arguments(parser)
    lrer.add_parser_arguments(parser)
    
    return parser


def run_script(config, proxy_file, proxy_class):
    # TorchTask requires PyTorch >= 1.0.0
    pytorch_support(required_version='1.0.0', info_str='TorchTask')

    # help information
    if len(sys.argv) > 1 and sys.argv[1] in ['help', '--help', 'h', '-h']:
        config['h'] = True

    # create parser and parse args from config
    parser = create_parser()
    proxy_file.add_parser_arguments(parser)
    args = cmd.parse_args(parser, config)

    task_proxy = proxy_class(args)
    task_proxy.run()
