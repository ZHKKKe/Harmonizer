
import torch
import torch.nn as nn

import torchtask


def add_parser_arguments(parser):
    torchtask.criterion_template.add_parser_arguments(parser)



def harmonizer_loss():
    return HarmonizerLoss


class AbsoluteLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(AbsoluteLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, gt):
        loss = torch.sqrt((pred - gt) ** 2 + self.epsilon)
        return loss


class HarmonizerLoss(torchtask.criterion_template.TaskCriterion):
    def __init__(self, args):
        super(HarmonizerLoss, self).__init__(args)

        self.l1 = AbsoluteLoss()
        self.l2 = nn.MSELoss(reduction='none')

    def forward(self, pred, gt, inp):
        pred_outputs, = pred
        x, mask = inp

        assert len(pred_outputs) == len(gt)

        image_losses = []
        for pred_, gt_ in zip(pred_outputs, gt):
            l1_loss = torch.sum(self.l1(pred_, gt_) * mask, dim=(1, 2, 3)) / (torch.sum(mask, dim=(1, 2, 3)) + 1e-6)
            l2_loss = torch.sum(self.l2(pred_, gt_) * mask, dim=(1, 2, 3)) / (torch.sum(mask, dim=(1, 2, 3)) + 1e-6) * 10
            loss = (l1_loss + l2_loss)
            image_losses.append(loss)

        return image_losses
