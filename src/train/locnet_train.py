import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial
from tqdm import tqdm
from argparse import ArgumentParser
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import random

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
# from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

from src.models import *
from src.data import LOCDataset, get_image_transform, get_depth_transform
from src.sl_utils import WandBLogger
from src import sl_utils

parser = ArgumentParser(description = 'Pytorch Imagenet Training')

parser.add_argument('--dataset', default = 'Imagenet')
parser.add_argument('--train_data_dir', help = 'Path to the train dataset')
parser.add_argument('--val_data_dir', help = 'Path to the val dataset')
parser.add_argument('--depth_dir', help = 'depth_maps')
parser.add_argument('--train_split', default = 'train', help = 'Train folder name')
parser.add_argument('--val_split', default = 'val', help = 'Validation folder name')
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--dataset_download', default = False)
parser.add_argument('--validation_batch_size', type = int, default = 128)
parser.add_argument('--checkpoint-hist', default = 10, type = int)
parser.add_argument('--resume', default = False, type = bool)
parser.add_argument('--epochs', default = 300, type = int)
parser.add_argument('--start-epoch', default = 0, type = int)
parser.add_argument('--loss_alpha', default = 0.5, type = float, help  = 'Loss importance for classification')

## model parameters
parser.add_argument('--model', help = 'name of the model', default = 'resnet50')
parser.add_argument('--pretrained', default = True)
parser.add_argument('--initial_checkpoint', default = None)
# parser.add_argument('--num_classes', default = 1000)
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                   help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--bn-momentum', type=float, default=None,
                   help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                   help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                   help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                   help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                   help='Enable separate BN layers per augmentation split.')
parser.add_argument('--grad-accum-steps', default = 1)
parser.add_argument('--log_interval', default = 100, type = int)

parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                   help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value", "agc")')
parser.add_argument('--layer-decay', type=float, default=None,
                   help='layer-wise learning rate decay (default: None)')
parser.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

## Learning rate parameters
parser.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "step"')
parser.add_argument('--sched-on-updates', action='store_true', default=False,
                   help='Apply LR scheduler step on update instead of epoch end.')
parser.add_argument('--lr', type=float, default=1.5e-3, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
parser.add_argument('--lr-base', type=float, default=0.4, metavar='LR',
                   help='base learning rate: lr = lr_base * global_batch_size / base_size')
parser.add_argument('--lr-base-size', type=int, default=128, metavar='DIV',
                   help='base learning rate batch size (divisor, default: 256).')
parser.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                   help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                   help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                   help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                   help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                   help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                   help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                   help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                   help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-4, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
parser.add_argument('--min-lr', type=float, default=1e-4, metavar='LR',
                   help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                    help='number of epochs to train (default: 300)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                   help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
# parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
#                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                   help='list of decay epoch indices for multistep lr. must be increasing')
parser.add_argument('--decay-epochs', type=float, default=10, metavar='N',
                   help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--warmup-prefix', action='store_true', default=False,
                   help='Exclude warmup period from decay schedule.'),
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                   help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                   help='patience epochs for Plateau LR scheduler (default: 10)')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')

# parser.add_argument('Augmentation and regularization parameters')
parser.add_argument('--no-aug', action='store_true', default=False,
                   help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                   help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                   help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                   help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                   help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                   help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-repeats', type=float, default=0,
                   help='Number of augmentation repetitions (distributed training only) (default: 0)')
parser.add_argument('--aug-splits', type=int, default=0,
                   help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd-loss', action='store_true', default=False,
                   help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--bce-loss', action='store_true', default=False,
                   help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument('--bce-target-thresh', type=float, default=None,
                   help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                   help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                   help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                   help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                   help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                   help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                   help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                   help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                   help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                   help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                   help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                   help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.01,
                   help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                   help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                   help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                   help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                   help='Drop block rate (default: None)')
parser.add_argument("--rank", default=0, type=int)
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                   help='disable fast prefetcher')
parser.add_argument('-j', '--workers', type=int, default=1, metavar='N',
                   help='how many training processes to use (default: 4)')
parser.add_argument('--amp', action='store_true', default=False,
                   help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
parser.add_argument('--amp-impl', default='native', type=str,
                   help='AMP impl to use, "native" or "apex" (default: native)')
parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                   help='Force broadcast buffers for native DDP to off.')
parser.add_argument('--synchronize-step', action='store_true', default=False,
                   help='torch.cuda.synchronize() end of each step')
parser.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--output', default='checkpoint', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                   help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                   help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--log-wandb', action='store_true', default=False,
                   help='log training and validation metrics to wandb')
parser.add_argument('--log-dir', type = str, default = 'logs', help = 'directory to store logs')


args = parser.parse_args()

_logger = logging.getLogger('train')


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

## train for one epoch
def train_one_epoch(model, start_epoch, train_dataloader, loss_cls, loss_recon, optimizer, device, lr_scheduler_values = [], lr_scheduler = None, log_writer = None):
    
    model.train()
    metric_logger = sl_utils.MetricLogger(delimiter =' ')
    header = 'TRAIN epoch: [{}]'.format(start_epoch)
    start_step = start_epoch*len(train_dataloader) + 1
    
    for i, (input,depth, target) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        input, depth, target = input.to(device), depth.to(device), target.to(device)
        output  = model(input)
        
        if isinstance(output, (tuple, list)):
            predicted_depth_map, predicted_class = output
    
        loss1 = loss_cls(predicted_class, target)
        loss2 = loss_recon(predicted_depth_map, depth)
        acc1, acc = sl_utils.accuracy(predicted_class, target, args, topk = (1,5))
        loss = args.loss_alpha * loss1 + (1 - args.loss_alpha) * loss2
        loss.backward()
 
        metric_logger.update(mse = loss2.item())
        metric_logger.update(cce = loss1.item())
        metric_logger.update(loss = loss.item())
        metric_logger.update(top1_accuracy = acc1.item())
        metric_logger.update(top5_accuracy = acc.item()) 
          
        if len(lr_scheduler_values)>0:
            for k, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_scheduler_values[start_step+i] #* param_group['lr_scale']
        
        optimizer.step() 
        metric_logger.synchronize_between_processes()

        if utils.is_primary(args) and log_writer!=None and ((start_step + i)%args.log_interval == 0):
            log_writer.set_step(start_step+i)

            log_writer.update(train_cce = metric_logger.cce.avg, head = 'train')
            log_writer.update(train_mse = metric_logger.mse.avg, head = 'train')
            log_writer.update(train_loss = metric_logger.loss.avg, head = 'train')
            log_writer.update(train_top1_accuracy = metric_logger.top1_accuracy.avg, head = 'train')
            log_writer.update(train_top5_accuracy = metric_logger.top5_accuracy.avg, head = 'train')
            log_writer.update(epoch = start_epoch, head = 'train')
            log_writer.update(commit=True, learning_rate = lr_scheduler_values[start_step+i], head = 'train')        
        
    return OrderedDict([('loss', metric_logger.loss.global_avg), ('top1', metric_logger.top1_accuracy.global_avg), ('top5', metric_logger.top5_accuracy.global_avg)])


def validate(model, start_epoch, val_dataloader, loss_fn, loss_recon, device, log_writer = None, start_step = 0):
    
    model.eval()
    metric_logger = sl_utils.MetricLogger(delimiter="  ")
    header = 'EVAL epoch: [{}]'.format(start_epoch)

    for i, (input,depth, target) in enumerate(tqdm(val_dataloader)):
        input, depth, target = input.to(device), depth.to(device), target.to(device)
        output  = model(input)
        if isinstance(output, (tuple, list)):
            predicted_depth_map, predicted_class = output
        
        loss1 = loss_fn(predicted_class, target)
        loss2 = loss_recon(predicted_depth_map, depth)
        loss = args.loss_alpha * loss1 + (1 - args.loss_alpha) * loss2
        
        acc1, acc = utils.accuracy(predicted_class, target, topk = (1,5))

        metric_logger.update(mse = loss2.item())
        metric_logger.update(cce = loss1.item())
        metric_logger.update(loss = loss.item())
        metric_logger.update(top1_accuracy = acc1.item())
        metric_logger.update(top5_accuracy = acc.item())
        metric_logger.synchronize_between_processes()
        
    if utils.is_primary(args) and log_writer!=None:
        log_writer.set_step(start_step)
        log_writer.update(val_cce = metric_logger.cce.global_avg, head = 'val')
        log_writer.update(val_mse = metric_logger.mse.global_avg, head = 'val')
        log_writer.update(val_loss = metric_logger.loss.global_avg, head = 'val')
        log_writer.update(val_top1_accuracy = metric_logger.top1_accuracy.global_avg, head = 'val')
        log_writer.update(val_top5_accuracy = metric_logger.top5_accuracy.global_avg, heaad = 'val')
        log_writer.update(commit=True, epoch = start_epoch, head = 'val')
    return OrderedDict([('loss', metric_logger.loss.global_avg), ('top1', metric_logger.top1_accuracy.global_avg), ('top5', metric_logger.top5_accuracy.global_avg)])


## main function
def main():

    num_epochs = args.epochs
    log_writer = None
    device = utils.init_distributed_device(args)
    if utils.is_primary(args):
        print(f"Is distributed training : {args.distributed}")
        if args.distributed:
            _logger.info(
                'Training in distributed mode with multiple processes, 1 device per process.'
                f'Process {args.rank}, total {args.world_size}, device {args.device}.')
        else:
            _logger.info(f'Training with a single process on 1 device ({args.device}).') 
        
        if args.log_wandb and args.log_dir != None:
            print("This else executes!")
            os.makedirs(args.log_dir, exist_ok = True)
            log_writer = WandBLogger(log_dir = args.log_dir , args = args)
        else:
            log_writer = None

    assert args.rank >= 0

    image_transformation = get_image_transform()
    depth_transformation = get_depth_transform()
    train_depth_dir = os.path.join(args.depth_dir, 'train')
    dataset_train = LOCDataset(args.train_data_dir, train_depth_dir, image_transform = image_transformation, depth_transform=depth_transformation)
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas = args.world_size, rank = args.rank, shuffle = True
    )

    val_depth_dir = os.path.join(args.depth_dir, 'val' )
    dataset_eval = LOCDataset(args.val_data_dir, val_depth_dir, image_transform = image_transformation, depth_transform= depth_transformation)
    sampler_eval = torch.utils.data.DistributedSampler(
        dataset_eval, num_replicas = args.world_size, rank = args.rank, shuffle = False
    )

    print(f"World Size: {args.world_size}")
    ## create model
    in_chans = 3

    ## load custom model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        # num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        # scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
    )
    model = model.to(device)

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    if args.distributed:
        if utils.is_primary(args):
            _logger.info("Using native Torch DistributedDataParallel.")
        model = NativeDDP(model, device_ids=[device], find_unused_parameters = False)

    loader_train = torch.utils.data.DataLoader(dataset_train, sampler = sampler_train, batch_size = args.batch_size)
    loader_eval = torch.utils.data.DataLoader(dataset_eval, sampler = sampler_eval, batch_size = args.batch_size)

    optimizer = create_optimizer_v2(
       model,
       **optimizer_kwargs(cfg=args),
       **args.opt_kwargs,
    )

    # optionally resume from a checkpoint
    loss_scaler = None
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # Can be made adaptable to BCE and other losses check pytorch_image_models repo
    if args.smoothing:
        train_cls_loss_fn = LabelSmoothingCrossEntropy(smoothing = args.smoothing).to(device)
    else:
        train_cls_loss_fn = nn.CrossEntropyLoss().to(device)
    train_recon_loss_fn = nn.MSELoss()
    
    validate_cls_loss_fn = nn.CrossEntropyLoss().to(device)
    validate_recon_loss_fn = nn.MSELoss()

    eval_metrics = "loss"
    model = model.to(device)

    saver = None
    if utils.is_primary(args):
        if args.output:
            os.makedirs(args.output, exist_ok=True)

        saver = utils.CheckpointSaver(
            model = model,
            optimizer = optimizer,
            args = args,
            checkpoint_dir = args.output,
            recovery_dir = args.output,
            decreasing = eval_metrics,
            max_history = args.checkpoint_hist
        )

    num_training_steps_per_epoch = len(loader_train)
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps

    lr_scheduler = None
    lr_scheduler_values = []
    lr_scheduler_values = sl_utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )

    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)
    
    for epoch in range(start_epoch, num_epochs):
        if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
        elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
            loader_train.sampler.set_epoch(epoch)

        if utils.is_primary(args) and  log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(model, epoch, loader_train, train_cls_loss_fn, train_recon_loss_fn, optimizer, device, lr_scheduler_values, lr_scheduler, log_writer)
        val_stats = validate(model, epoch, loader_eval, validate_cls_loss_fn, validate_recon_loss_fn, device, log_writer, ((epoch+1)*num_training_steps_per_epoch)+1)

        if utils.is_primary(args) and saver is not None:
            best_metric, best_epoch = saver.save_checkpoint(epoch, metric = val_stats['loss'])

        if utils.is_primary(args) and log_writer is not None:
            log_writer.flush()
    
    if utils.is_primary(args):
        print(best_metric, best_epoch)


if __name__ == '__main__':
    main()
