import os
import time
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

import torchtask
from torchtask.utils import logger, cmd, tool
from torchtask.nn import func


def add_parser_arguments(parser):
    torchtask.trainer_template.add_parser_arguments(parser)


def harmonizer_trainer(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    model_funcs = [model_dict['model']]
    optimizer_funcs = [optimizer_dict['model']]
    lrer_funcs = [lrer_dict['model']]
    criterion_funcs = [criterion_dict['model']]

    algorithm = HarmonizerTrainer(args)
    algorithm.build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)
    return algorithm


class HarmonizerTrainer(torchtask.trainer_template.TaskTrainer):
    def __init__(self, args):
        super(HarmonizerTrainer, self).__init__(args)

        self.model = None
        self.optimizer = None
        self.lrer = None
        self.criterion = None

    def _build(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        self.task_func = task_func

        self.model = func.create_model(model_funcs[0], 'model', args=self.args)
        self.models = {'model': self.model}

        self.optimizer = optimizer_funcs[0](self.model.module.param_groups)
        self.optimizers = {'optimizer': self.optimizer}

        self.lrer = lrer_funcs[0](self.optimizer)
        self.lrers = {'lrer': self.lrer}

        self.criterion = criterion_funcs[0](self.args)
        self.criterions = {'criterion': self.criterion}

    def _train(self, data_loader, epoch):
        self.meters.reset()

        lbs = self.args.labeled_batch_size

        self.model.train()

        timer = time.time()
        for idx, (inp, gt) in enumerate(data_loader):
            # pre-process input tensor and ground truth tensor
            inp, gt = self._batch_prehandle(inp, gt, True)
            x, mask = inp

            # forword the model
            self.optimizer.zero_grad()
            resulter, debugger = self.model(inp)

            pred_outputs = tool.dict_value(resulter, 'outputs')

            # calculate loss for the fine labeled data
            l_pred_outputs = func.split_tensor_tuple(pred_outputs, 0, lbs)
            l_pred = (l_pred_outputs, )

            l_gt = func.split_tensor_tuple(gt, 0, lbs)
            l_inp = func.split_tensor_tuple(inp, 0, lbs)

            l_image_losses = self.criterion(l_pred, l_gt, l_inp)

            # if self.args.dynamic_loss:
            sum_losses = l_image_losses[0].detach()
            for i in range(1, len(l_image_losses)):
                sum_losses = sum_losses + \
                    (l_image_losses[i].detach() - l_image_losses[i-1].detach()) * ((l_image_losses[i].detach() - l_image_losses[i-1].detach()) > 0).float()
            sum_losses = sum_losses + 1e-9
            sum_losses = sum_losses.detach()

            scaled_l_image_losses = [torch.mean(l_image_losses[0] / sum_losses)]
            self.meters.update('fine_filter_0_loss', torch.mean(l_image_losses[0] / sum_losses).item())

            for i in range(1, len(l_image_losses)):
                loss = (l_image_losses[i] - l_image_losses[i-1].detach()) / sum_losses
                loss = loss * (loss > 0).float()
                loss = torch.mean(loss)
                scaled_l_image_losses.append(loss)
                self.meters.update('fine_filter_{0}_loss'.format(i), loss.item())
            
            # calculate loss for the coarse labeled data
            if not self.args.ignore_additional:
                u_pred_outputs = func.split_tensor_tuple(pred_outputs, lbs, self.args.batch_size)
                u_pred_outputs = (u_pred_outputs[-1], )
                u_pred = (u_pred_outputs, )

                u_gt = func.split_tensor_tuple(gt, lbs, self.args.batch_size)
                u_gt = (u_gt[-1], )

                u_inp = func.split_tensor_tuple(inp, lbs, self.args.batch_size)
                
                u_image_losses = self.criterion(u_pred, u_gt, u_inp)

                u_image_loss = torch.mean(u_image_losses[0]) * 10

                self.meters.update('coarse_filter_loss', u_image_loss.item())
            else:
                self.meters.update('coarse_filter_loss', torch.mean(torch.zeros(1)).item())

            # calculate the sum of all losses
            loss = 0
            for l_image_loss in scaled_l_image_losses:
                loss = loss + l_image_loss
            loss = loss + u_image_loss

            # backward and update
            loss.backward()
            self.optimizer.step()

            # logging
            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}'.format(epoch+1, idx, len(data_loader), meters=self.meters))
                logger.log_info('\tfine-filter-0-loss: {meters[fine_filter_0_loss]:.6f}'.format(meters=self.meters))
                logger.log_info('\tfine-filter-1-loss: {meters[fine_filter_1_loss]:.6f}'.format(meters=self.meters))
                logger.log_info('\tfine-filter-2-loss: {meters[fine_filter_2_loss]:.6f}'.format(meters=self.meters))
                logger.log_info('\tfine-filter-3-loss: {meters[fine_filter_3_loss]:.6f}'.format(meters=self.meters))
                logger.log_info('\tfine-filter-4-loss: {meters[fine_filter_4_loss]:.6f}'.format(meters=self.meters))
                logger.log_info('\tfine-filter-5-loss: {meters[fine_filter_5_loss]:.6f}'.format(meters=self.meters))
                logger.log_info('\tcoarse-filter-loss: {meters[coarse_filter_loss]:.6f}'.format(meters=self.meters))

            # visualization
            if self.args.visualize and idx % self.args.visual_freq == 0:
                self._visualization(
                    epoch, idx, True,
                    func.split_tensor_tuple(inp, 0, 1, reduce_dim=True),
                    func.split_tensor_tuple(pred_outputs, 0, 1, reduce_dim=True),
                    func.split_tensor_tuple(gt, 0, 1, reduce_dim=True))

            # update iteration-based lrers
            if not self.args.is_epoch_lrer:
                self.lrer.step()

            timer = time.time()

        # update epoch-based lrers
        if self.args.is_epoch_lrer:
            self.lrer.step()

    def _validate(self, data_loader, epoch):
        self.meters.reset()

        self.model.eval()

        timer = time.time()
        for idx, (inp, gt) in enumerate(data_loader):
            inp, gt = self._batch_prehandle(inp, gt, False)
            x, mask = inp

            resulter, debugger = self.model(inp)

            pred_outputs = tool.dict_value(resulter, 'outputs')

            pred = (pred_outputs[-1], )
            gt = (gt[-1], )

            # calculate loss for the fine labeled data
            losses = self.criterion.forward(pred, gt, inp)
            loss = 0
            for _loss in losses:
                loss = loss + _loss
            loss = loss / len(losses)

            self.meters.update('loss', loss.item())

            self.task_func.metrics(pred_outputs[-1].detach(), gt[-1], mask, self.meters, id_str='IH')    

            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                'loss: {meters[loss]:.6f}\n'
                                .format(epoch+1, idx, len(data_loader), meters=self.meters))

            if self.args.visualize:
                self._visualization(
                    epoch, idx, False,
                    func.split_tensor_tuple(inp, 0, 1, reduce_dim=True),
                    func.split_tensor_tuple((pred_outputs[-1], ), 0, 1, reduce_dim=True),
                    func.split_tensor_tuple(gt, 0, 1, reduce_dim=True))

            timer = time.time()

        metrics_info = {'IH': ''}
        for key in sorted(list(self.meters.keys())):
            if self.task_func.METRIC_STR in key:
                for id_str in metrics_info.keys():
                    if key.startswith(id_str):
                        metrics_info[id_str] += '{0}: {1:.6}\t'.format(key, self.meters[key])
            
        logger.log_info('Validation metrics:\n task-metrics\t=>\t{0}\n'.format(metrics_info['IH'].replace('_', '-')))

    def _visualization(self, epoch, idx, is_train, inp, pred, gt):
        visualize_path = self.args.visual_train_path if is_train else self.args.visual_val_path
        out_path = os.path.join(visualize_path, '{0}_{1}'.format(epoch, idx))

        x, mask = inp

        x = (np.transpose(x.cpu().numpy(), (1, 2, 0)))
        Image.fromarray((x * 255).astype('uint8')).save(out_path + '_1_0_x.jpg')

        mask = mask[0].data.cpu().numpy()
        Image.fromarray((mask * 255).astype('uint8'), mode='L').save(out_path + '_2_0_mask.jpg')

        for idx, (pred_, gt_) in enumerate(zip(pred, gt)):
            pred_ = (np.transpose(pred_.detach().cpu().numpy(), (1, 2, 0)))
            Image.fromarray((pred_ * 255).astype('uint8')).save(out_path + '_1_{0}_pred_filter.jpg'.format(idx+1))

            if torch.mean(gt_) != -999:
                gt_ = (np.transpose(gt_.cpu().numpy(), (1, 2, 0)))
                Image.fromarray((gt_ * 255).astype('uint8')).save(out_path + '_2_{0}_gt_filter.jpg'.format(idx+1))

    def _save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lrer': self.lrer.state_dict(),
        }
        checkpoint = os.path.join(self.args.checkpoint_path, 'checkpoint_{0}.ckpt'.format(epoch))

        torch.save(state, checkpoint)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.args.resume)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lrer.load_state_dict(checkpoint['lrer'])
        return checkpoint['epoch']

    def _batch_prehandle(self, inp, gt, is_train):
        lbs = self.args.labeled_batch_size
        ubs = self.args.additional_batch_size

        # convert all input and ground truth to Variables
        inp_var = []
        for i in inp:
            inp_var.append(Variable(i).cuda())
        inp = tuple(inp_var)
       
        gt_var = []
        for g in gt:
            gt_var.append(Variable(g).cuda())
        gt = tuple(gt_var)

        filter_num = len(self.model.module.model.filter_types)

        if is_train:
            # ----------------------------------------------------------------
            # for fine labeled data, we generate the adjusted input
            # ----------------------------------------------------------------
            l_inp = func.split_tensor_tuple(inp, 0, lbs)
            l_gt = func.split_tensor_tuple(gt, 0, lbs)

            _, l_mask = l_inp
            l_gt_image, = l_gt

            n = l_gt_image.shape[0]
            l_rand_arguments = [self._rand_adjustment_values(n) for _ in range(0, filter_num)]

            l_x = self.model.module.adjust(l_gt_image, l_mask, l_rand_arguments)

            l_inp = (l_x[-1], l_mask)
            l_gt = []
            for _ in reversed(l_x[:-1]):
                l_gt.append(_)
            l_gt.append(l_gt_image)

            if not self.args.ignore_additional:
                # ----------------------------------------------------------------
                # for coarse labeled data, we use the existising adjusted input
                # ----------------------------------------------------------------
                u_inp = func.split_tensor_tuple(inp, lbs, self.args.batch_size)
                u_gt = func.split_tensor_tuple(gt, lbs, self.args.batch_size)

                u_gt_image, = u_gt
                none_value = torch.ones(ubs).view(ubs, 1).cuda() * -999
                none_im = u_gt_image.cuda() * 0 - 999

                u_gt = [none_im for _ in range(0, filter_num)]
                u_gt[-1] = u_gt_image

                inp = func.combine_tensor_tuple(l_inp, u_inp, 0)
                gt = func.combine_tensor_tuple(l_gt, u_gt, 0)
                
            else:
                inp = l_inp
                gt = l_gt

        else:
            gt_image, = gt

            none_value = torch.ones(1).view(1, 1).cuda() * -999
            none_im = gt_image.cuda() * 0 - 999

            gt = [none_im for _ in range(0, filter_num)]
            gt[-1] = gt_image

        return inp, gt

    def _rand_adjustment_values(self, n):
        x = torch.FloatTensor(np.random.uniform(-1, 1, n))
        x = x.view(n, 1).cuda()
        return x
    