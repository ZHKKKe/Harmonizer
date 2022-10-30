import torchtask

import func, data, model, criterion, trainer


def add_parser_arguments(parser):
    torchtask.proxy_template.add_parser_arguments(parser)
    
    data.add_parser_arguments(parser)
    model.add_parser_arguments(parser)
    criterion.add_parser_arguments(parser)
    trainer.add_parser_arguments(parser)


class HarmonizerProxy(torchtask.proxy_template.TaskProxy):

    NAME = 'harmonizer'

    def __init__(self, args):
        super(HarmonizerProxy, self).__init__(args, func, data, model, criterion, trainer)
