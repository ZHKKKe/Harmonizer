import os
import time
import yaml
import copy
from datetime import datetime

import torch

import torchtask
from torchtask.utils import logger, cmd
from torchtask.nn import data as nndata
from torchtask.nn import lrer as nnlrer
from torchtask.nn import optimizer as nnoptimizer


def add_parser_arguments(parser):
    # experimental arguments
    parser.add_argument('--exp-id', type=str, default='', metavar='', help='exp - the unique id (or name) of experiment')
    parser.add_argument('--resume', type=str, default='', metavar='', help='exp - the checkpoint file that will be resumed')
    parser.add_argument('--validation', type=cmd.str2bool, default=False, metavar='', help='exp - validation only if True')
    parser.add_argument('--out-path', type=str, default='', metavar='', help='exp - the path where the output of experiment is stored')
    parser.add_argument('--visualize', type=cmd.str2bool, default=False, metavar='', help='exp - save the output images for visualization if True')  
    parser.add_argument('--debug', type=cmd.str2bool, default=False, metavar='', help='exp - experiment under debug mode if True')
    parser.add_argument('--val-freq', type=int, default=1, metavar='', help='exp - validation frequency during training [unit: epoch]')
    parser.add_argument('--log-freq', type=int, default=100, metavar='', help='exp - logging frequency during training and validation [unit: iteration]')
    parser.add_argument('--visual-freq', type=int, default=100, metavar='', help='exp - visulization frequency during training and validation [unit: iteration]')
    parser.add_argument('--checkpoint-freq', type=int, default=1, metavar='', help='exp - checkpoint saving frequency during training [unit: epoch]')

    # dataset / dataloader arguments
    parser.add_argument('--trainset', type=yaml.full_load, default={}, metavar='', help='data - path of the train dataset of format {dataset_type: [path1, path2]}')
    parser.add_argument('--valset', type=yaml.full_load, default={}, metavar='', help='data - path of the validate dataset of format {dataset_type: [path1, path2]}')
    parser.add_argument('--num-workers', type=int, default=1, metavar='', help='data - number of workers for the data loader on each GPU')
    parser.add_argument('--im-size', type=int, default=None, help='data - target size of the input images')
    parser.add_argument('--additionalset', type=yaml.full_load, default={}, metavar='', help='data - path of the extra additional dataset of format {dataset_type: [path1, path2]}')
    parser.add_argument('--sublabeled-path', type=str, default='', metavar='', help='data - path of the file that stores the prefix of the labeled subset')
    parser.add_argument('--ignore-additional', type=cmd.str2bool, default=True, metavar='', help='data - ignore (do not use) the additional samples during training if True')
    parser.add_argument('--short-ep', type=cmd.str2bool, default=False, metavar='', help='data - ')

    # task algorithm arguments
    parser.add_argument('--trainer', type=str, default='', metavar='', help='task - the task algorithm used in experiment')
    parser.add_argument('--models', type=yaml.full_load, default={}, metavar='', help='task - dict saves all {component_key: task_model} for the task algorithm')
    parser.add_argument('--optimizers', type=yaml.full_load, default={}, metavar='', help='task - dict saves all {component_key: task_optimizer} for the task algorithm')
    parser.add_argument('--lrers', type=yaml.full_load, default={}, metavar='', help='task - dict saves all {component_key: task_lrer} for the task algorithm')
    parser.add_argument('--criterions', type=yaml.full_load, default={}, metavar='', help='task - dict saves all {componet_key: task_criterion} for the task algorithm')

    # training arguemnts
    parser.add_argument('--epochs', type=int, default=1, metavar='', help='train/val - total epochs for training')
    parser.add_argument('--batch-size', type=int, default=16, metavar='', help='train/val - total batch size for training/validation on each GPU')
    parser.add_argument('--additional-batch-size', type=int, default=0, metavar='', help='train/val - number of additional samples in a mini-batch on each GPU')

    # arguments set by the code of proxy
    parser.add_argument('--gpus', type=int, default=0, metavar='', help='autoset - number of GPUs for running [this argument is automatically set by code!]')
    parser.add_argument('--task', type=str, default='', metavar='', help='autoset - name string of current task [this argument is automatically set by code!]')
    parser.add_argument('--labeled-batch-size', type=int, default=None, metavar='', help='autoset - number of labeled samples in a mini-batch on each GPU [this argument is automatically set by code!]')
    parser.add_argument('--checkpoint-path', type=str, default='', metavar='', help='autoset - the path used to save the checkpoint files [this argument is automatically set by code!]')
    parser.add_argument('--visual-debug-path', type=str, default='', metavar='', help='autoset - the path used to save the debuging images for visualization [this argument is automatically set by code!]') 
    parser.add_argument('--visual-train-path', type=str, default='', metavar='', help='autoset - the path used to save the training images for visualization [this argument is automatically set by code!]') 
    parser.add_argument('--visual-val-path', type=str, default='', metavar='', help='autoset - the path used to save the validation images for visualization [this argument is automatically set by code!]')
    parser.add_argument('--is-epoch-lrer', type=cmd.str2bool, default=None, metavar='', help='autoset - adjust the learning rate after (1) each epoch (if True) or each iter (if False) [this argument is automatically set by code!]')
    parser.add_argument('--iters-per-epoch', type=int, default=None, metavar='', help='autoset - number of iterations inside each epoch [this argument is automatically set by code!]')


class TaskProxy:
    NAME = 'task'                       # specific name of task

    def __init__(self, args, func, data, model, criterion, trainer):
        self.args = args                # arguments dict for task-specific proxy

        self.func = func                # instance of 'TaskFunc'
        self.data = data                # instance of 'TaskData'
        self.model = model              # instance of 'TaskModel'
        self.criterion = criterion      # instance of 'TaskCriterion'

        self.trainer = None
        self.trainer_class = trainer
        self.model_dict = {}
        self.criterion_dict = {}
        self.optimizer_dict = {}
        self.lrer_dict = {}

        self.train_loader = None        # instance of 'torch.utils.data.DataLoader'
        self.val_loader = None          # instance of 'torch.utils.data.DataLoader'

        self._init()

    def run(self):
        self._run()

    def _run(self):
        start_epoch = 0
        if self.args.resume is not None and self.args.resume != '':
            logger.log_info('Load checkpoint from: {0}'.format(self.args.resume))
            start_epoch = self.trainer.load_checkpoint()
        
        if self.args.validation:
            if self.val_loader is None:
                logger.log_err('No data loader for validation.\n'
                               'Please set right \'valset\' in the script.\n')
                        
            logger.log_info(['=' * 78, '\nStart to validate model\n', '=' * 78])
            with torch.no_grad():
                self.trainer.validate(self.val_loader, start_epoch - 1)

            self.trainer.save_checkpoint(0)
            return

        # NOTE: the first epoch index for 'train' and 'validatie' is 0
        for epoch in range(start_epoch, self.args.epochs):
            timer = time.time()

            logger.log_info(['=' * 78, '\nStart to train epoch-{0}\n'.format(epoch + 1), '=' * 78])
            self.trainer.train(self.train_loader, epoch)

            if (epoch + 1) % self.args.val_freq == 0 and self.val_loader is not None:
                logger.log_info(['=' * 78, '\nStart to validate epoch-{0}\n'.format(epoch + 1), '=' * 78])
                with torch.no_grad():
                    self.trainer.validate(self.val_loader, epoch)

            if (epoch + 1) % self.args.checkpoint_freq == 0:
                self.trainer.save_checkpoint(epoch + 1)
                logger.log_info("Save checkpoint for epoch {0}".format(epoch + 1))
        
            logger.log_info('Finish epoch in {0} seconds\n'.format(time.time() - timer))
        
        logger.log_info('Finish experiment {0}\n'.format(self.args.exp_id))

    def _init(self):
        """ Initial function of the task proxy.
        """

        self._preprocess_arguments()
        self._create_dataloader()
        self._build_trainer()    

    def _preprocess_arguments(self):
        """ Preprocess the arguments in the script.
        """

        # create the output folder to store the results
        self.args.out_path = "{root}/{exp_id}/{date:%Y-%m-%d_%H-%M-%S}/".format(
            root=self.args.out_path, exp_id=self.args.exp_id, date=datetime.now())
        if not os.path.exists(self.args.out_path):
            os.makedirs(self.args.out_path)

        # prepare logger
        exp_op = 'val' if self.args.validation else 'train'
        logger.log_mode(self.args.debug)
        logger.log_file(os.path.join(self.args.out_path, '{0}.log'.format(exp_op)), self.args.debug)

        logger.log_info('Result folder: \n  {0} \n'.format(self.args.out_path))

        # print experimental args
        cmd.print_args()

        # set task name
        self.args.task = self.NAME

        # check the task-specific components dicts required by the task algorithm
        if not len(self.args.models) == len(self.args.optimizers) == len(self.args.lrers) == len(self.args.criterions):
            logger.log_err('Condition:\n'
                           '\tlen(self.args.models) == len(self.args.optimizers) == len(self.args.lrers) == len(self.args.criterions\n'
                           'is not satisfied in the script\n')
        
        for (model, criterion, optimizer, lrer) in \
            zip(self.args.models.values(), self.args.criterions.values(), self.args.optimizers.values(), self.args.lrers.values()):
            if model not in self.model.__dict__:
                logger.log_err('Unsupport model: {0} for task: {1}\n'
                               'Please add the export function in task\'s \'model.py\'\n'.format(model, self.args.task))
            elif criterion not in self.criterion.__dict__:
                logger.log_err('Unsupport criterion: {0} for task: {1}\n'
                               'Please add the export function in task\'s \'criterion.py\'\n'.format(criterion, self.args.task))
            elif optimizer not in nnoptimizer.__dict__:
                logger.log_err('Unsupport optimizer: {0}\n'
                               'Please implement the optimizer wrapper in \'torchtask/nn/optimizer.py\'\n'.format(optimizer))
            elif lrer not in nnlrer.__dict__:
                logger.log_err('Unsupport learning rate scheduler: {0}\n'
                               'Please implement lr scheduler wrapper in \'torchtask/nn/lrer.py\'\n'.format(lrer))
            
        # check the types of lrers
        for lrer in self.args.lrers.values():
            if lrer in nnlrer.EPOCH_LRERS:
                is_epoch_lrer = True 
            elif lrer in nnlrer.ITER_LRERS:
                is_epoch_lrer = False
            else:
                logger.log_err('Unknown learning rate scheduler ({0}) type\n'
                               'Please add it into either EPOCH_LRERS or ITER_LRERS in \'torchtask/nn/lrer.py\'\n'
                               'TorchTask supports: \n'
                               '  EPOCH_LRERS\t=>\t{1}\n  ITER_LRERS\t=>\t{2}\n'.format(lrer, nnlrer.EPOCH_LRERS, nnlrer.ITER_LRERS))

            if self.args.is_epoch_lrer is None:
                self.args.is_epoch_lrer = is_epoch_lrer
            elif self.args.is_epoch_lrer != is_epoch_lrer:
                logger.log_err('Unmatched lr scheduler types\t=>\t{0}\n'
                               'All lrers of the task models should have the same types (either EPOCH_LRERS or ITER_LRERS)\n'
                               'TorchTask supports: \n'
                               '  EPOCH_LRERS\t=>\t{1}\n  ITER_LRERS\t=>\t{2}\n'
                               .format(self.args.lrers, nnlrer.EPOCH_LRERS, nnlrer.ITER_LRERS))

        self.args.checkpoint_path = os.path.join(self.args.out_path, 'ckpt')
        if not os.path.exists(self.args.checkpoint_path):
            os.makedirs(self.args.checkpoint_path)

        if self.args.visualize:
            self.args.visual_debug_path = os.path.join(self.args.out_path, 'visualization/debug')
            self.args.visual_train_path = os.path.join(self.args.out_path, 'visualization/train')
            self.args.visual_val_path = os.path.join(self.args.out_path, 'visualization/val')
            for vpath in [self.args.visual_debug_path, self.args.visual_train_path, self.args.visual_val_path]:
                if not os.path.exists(vpath):
                    os.makedirs(vpath)
        
        # handle argumens for multiply GPUs training
        self.args.gpus = torch.cuda.device_count()
        if self.args.gpus < 1:
            logger.log_err('No GPU be detected\n'
                           'TorchTask requires at least one Nvidia GPU\n')

        logger.log_info('GPU: \n  Total GPU(s): {0}'.format(self.args.gpus))
        self.args.lr *= self.args.gpus
        self.args.num_workers *= self.args.gpus
        self.args.batch_size *= self.args.gpus
        self.args.additional_batch_size *= self.args.gpus
        
        # TODO: support unsupervised and self-supervised training
        if self.args.additional_batch_size >= self.args.batch_size:
            logger.log_err('The argument \'additional_batch_size\' ({0}) should be smaller than \'batch_size\' ({1}) '
                           'since TorchTask only supports semi-supervised learning now\n')

        self.args.labeled_batch_size = self.args.batch_size - self.args.additional_batch_size
        logger.log_info('  Total learn rate: {0}\n  Total labeled batch size: {1}\n'
                        '  Total additional batch size: {2}\n  Total data workers: {3}\n'.format(
                    self.args.lr, self.args.labeled_batch_size, self.args.additional_batch_size, self.args.num_workers))

    def _create_dataloader(self):
        """ Create data loaders for experiment.
        """

        # ---------------------------------------------------------------------
        # create dataloder for training
        # ---------------------------------------------------------------------

        # ignore_additional == False & additional_batch_size != 0 
        #   means that both labeled and additional data are used
        with_additional_data = not self.args.ignore_additional and self.args.additional_batch_size != 0
        # ignore_additional == True & additional_batch_size == 0
        #   means that only the labeled data is used
        without_additional_data = self.args.ignore_additional and self.args.additional_batch_size == 0

        labeled_train_samples, additional_train_samples = 0, 0
        if not self.args.validation:
            # ignore_additional == True & additional_batch_size != 0 -> error
            if self.args.ignore_additional and self.args.additional_batch_size != 0:
                logger.log_err('Arguments conflict => ignore_additional == True requires additional_batch_size == 0\n')
            # ignore_additional == False & additional_batch_size == 0 -> error
            if not self.args.ignore_additional and self.args.additional_batch_size == 0:
                logger.log_err('Arguments conflict => ignore_additional == False requires additional_batch_size != 0\n')

            # calculate the number of trainsets
            trainset_num = 0
            for key, value in self.args.trainset.items():
                trainset_num += len(value)

            # calculate the number of additionalsets
            additionalset_num = 0
            for key, value in self.args.additionalset.items():
                additionalset_num += len(value) 

            # if only one labeled training set and without any additional set
            if trainset_num == 1 and additionalset_num == 0:
                trainset = self._load_dataset(list(self.args.trainset.keys())[0], list(self.args.trainset.values())[0][0])
                labeled_train_samples = len(trainset.idxs)

                # if the 'sublabeled_path' is given
                sublabeled_prefix = None
                if self.args.sublabeled_path is not None and self.args.sublabeled_path != '':
                    if not os.path.exists(self.args.sublabeled_path):
                        logger.log_err('Cannot find labeled file: {0}\n'.format(self.args.sublabeled_path))
                    else:
                        with open(self.args.sublabeled_path) as f:
                            sublabeled_prefix = [line.strip() for line in f.read().splitlines()]
                        sublabeled_prefix = None if len(sublabeled_prefix) == 0 else sublabeled_prefix

                if sublabeled_prefix is not None:
                    # wrap the trainset by 'SplitUnlabeledWrapper'
                    trainset = nndata.SplitUnlabeledWrapper(
                        trainset, sublabeled_prefix, ignore_additional=self.args.ignore_additional)
                    labeled_train_samples = len(trainset.labeled_idxs)
                    additional_train_samples = len(trainset.additional_idxs)

                # if 'sublabeled_prefix' is None but you want to use the additional data for training
                elif with_additional_data:
                    logger.log_err('Try to use the additional samples without any task dataset wrapper\n')

            # if more than one labeled training sets are given or the additional training sets are given
            elif trainset_num > 1 or additionalset_num > 0:
                # 'arg.sublabeled_path' is disabled
                if self.args.sublabeled_path is not None and self.args.sublabeled_path != '':
                    logger.log_err('Multiple training datasets are given. \n'
                                   'Inter-split additional set is not allowed.\n'
                                   'Please remove the argument \'sublabeled_path\' in the script\n')
                
                # load all training sets
                labeled_sets = []
                for set_name, set_dirs in self.args.trainset.items():
                    for set_dir in set_dirs:
                        labeled_sets.append(self._load_dataset(set_name, set_dir))
                    
                # load all extra additional sets
                additional_sets = []
                # if any extra additional set is given
                if additionalset_num > 0:
                    for set_name, set_dirs in self.args.additionalset.items():
                        for set_dir in set_dirs:
                            additional_sets.append(self._load_dataset(set_name, set_dir))

                # if unalbeledset_num == 0 but you want to use the additional data for training    
                elif with_additional_data:
                    logger.log_err('Try to use the additional samples without any task dataset wrapper\n'
                                   'Please add the argument \'additionalset\' in the script\n')

                # wrap both 'labeled_set' and 'additional_set' by 'JointDatasetsWrapper'
                trainset = nndata.JointDatasetsWrapper(
                    labeled_sets, additional_sets, ignore_additional=self.args.ignore_additional)
                labeled_train_samples = len(trainset.labeled_idxs)
                additional_train_samples = len(trainset.additional_idxs)

            # if use labeled data only
            if without_additional_data:
                self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, 
                    shuffle=True, num_workers=self.args.num_workers, pin_memory=True, drop_last=True)
            # if use both labeled and additional data
            elif with_additional_data:
                train_sampler = nndata.TwoStreamBatchSampler(trainset.labeled_idxs, trainset.additional_idxs, 
                    self.args.labeled_batch_size, self.args.additional_batch_size, short_ep=self.args.short_ep)
                self.train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=train_sampler, 
                    num_workers=self.args.num_workers, pin_memory=True)

        # ---------------------------------------------------------------------
        # create dataloader for validation
        # ---------------------------------------------------------------------

        # calculate the number of valsets
        valset_num = 0
        for key, value in self.args.valset.items():
            valset_num += len(value)

        # if only one validation set is given
        if valset_num == 1:
            valset = self._load_dataset(
                list(self.args.valset.keys())[0], list(self.args.valset.values())[0][0], is_train=False)
            val_samples = len(valset.idxs)

        # if more than one validation sets are given
        elif valset_num > 1:
            valsets = []
            for set_name, set_dirs in self.args.valset.items():
                for set_dir in set_dirs:
                    valsets.append(self._load_dataset(set_name, set_dir, is_train=False))
            valset = nndata.JointDatasetsWrapper(valsets, [], ignore_additional=True)
            val_samples = len(valset.labeled_idxs)
        
        # NOTE: batch size is set to 1 during the validation
        self.val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=True)

        # check the data loaders
        if self.train_loader is None and not self.args.validation:
            logger.log_err('Train data loader is required if validate mode is closed\n')
        elif self.val_loader is None and self.args.validation:
            logger.log_err('Validate data loader is required if validate mode is opened\n')
        elif self.val_loader is None:
            logger.log_warn('No validate data loader, there are no validation during the training\n')
        
        # set 'iters_per_epoch', which is required by ITER_LRERS
        self.args.iters_per_epoch = len(self.train_loader) if self.train_loader is not None else -1

        logger.log_info('Dataset:\n'
                        '  Trainset\t=>\tlabeled samples = {0}\t\tadditional samples = {1}\n'
                        '  Valset\t=>\tsamples = {2}\n'
                        .format(labeled_train_samples, additional_train_samples, val_samples))

    def _build_trainer(self):
        """ Build the semi-supervised learning algorithm given in the script.
        """

        for cname in self.args.models.keys():
            self.model_dict[cname] = self.model.__dict__[self.args.models[cname]]()
            self.criterion_dict[cname] = self.criterion.__dict__[self.args.criterions[cname]]()
            self.lrer_dict[cname] = nnlrer.__dict__[self.args.lrers[cname]](self.args)
            self.optimizer_dict[cname] = nnoptimizer.__dict__[self.args.optimizers[cname]](self.args)
        
        logger.log_info('Trainer: \n  {0}\n'.format(self.args.trainer))
        logger.log_info('Models: ')
        self.trainer = self.trainer_class.__dict__[self.args.trainer](
            self.args, self.model_dict, self.optimizer_dict, self.lrer_dict, self.criterion_dict, self.func.task_func()(self.args))

    def _load_dataset(self, dataset_name, dataset_dir, is_train=True):
        """ Load one dataset.
        """

        if not dataset_name in self.data.__dict__.keys():
            logger.log_err('Unknown dataset type: {0}\n'.format(dataset_name))
        elif not os.path.exists(dataset_dir):
            logger.log_err('Cannot find the path of dataset: {0}\n'.format(dataset_dir))
        else:
            dataset_args = copy.deepcopy(self.args)
            if is_train:
                dataset_args.trainset = {dataset_name: dataset_dir}
            else:
                dataset_args.valset = {dataset_name: dataset_dir}
            return self.data.__dict__[dataset_name]()(dataset_args, is_train)
