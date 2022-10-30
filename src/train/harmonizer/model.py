import torch
import torch.nn.functional as F

import torchtask

from module import harmonizer as _harmonizer


def add_parser_arguments(parser):
    torchtask.model_template.add_parser_arguments(parser)


def harmonizer():
    return Harmonizer


class Harmonizer(torchtask.model_template.TaskModel):
    def __init__(self, args):
        super(Harmonizer, self).__init__(args)

        self.model = _harmonizer.Harmonizer()
        self.param_groups = [
            {'params': filter(lambda p:p.requires_grad, self.model.backbone.parameters()), 'lr': self.args.lr},
            {'params': filter(lambda p:p.requires_grad, self.model.regressor.parameters()), 'lr': self.args.lr},
            {'params': filter(lambda p:p.requires_grad, self.model.performer.parameters()), 'lr': self.args.lr},
        ]
    
    def forward(self, inp):
        resulter, debugger = {}, {}
        x, mask = inp
        pred = self.model(x, mask)
        resulter['outputs'] = pred
        return resulter, debugger

    def restore(self, x, mask, arguments):
        with torch.no_grad():
            return self.model.restore_image(x, mask, arguments)

    def adjust(self, x, mask, arguments):
        with torch.no_grad():
            return self.model.adjust_image(x, mask, arguments)
