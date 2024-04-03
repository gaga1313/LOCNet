import logging
import os
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import numpy as np
import random
import cv2
from timm import utils
from timm.loss import LabelSmoothingCrossEntropy
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from src.data import LOCDataset, get_image_transform, get_depth_transform
from src.sl_utils import WandBLogger
from src import sl_utils

from src.utils.args import args

_logger = logging.getLogger("train")


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


# train for one epoch
def train_one_epoch(
    model,
    start_epoch,
    train_dataloader,
    loss_cls,
    loss_recon,
    optimizer,
    rest_cce,
    device,
    lr_scheduler_values=[],
    loss_annealing=[],
    lr_scheduler=None,
    log_writer=None,
):

    model.train()
    metric_logger = sl_utils.MetricLogger(delimiter=" ")
    header = "TRAIN epoch: [{}]".format(start_epoch)
    start_step = start_epoch * len(train_dataloader) + 1
    anneal_step = (start_epoch - rest_cce) * len(train_dataloader)
    if anneal_step < 0:
        anneal_step = 0

    if start_epoch >= rest_cce:
        init_cce = 1.0
    else:
        init_cce = 0.0

    for i, (input, depth, target) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        input, depth, target = input.to(device), depth.to(device), target.to(device)
        output = model(input)

        if isinstance(output, (tuple, list)):
            predicted_depth_map, predicted_class = output

        loss1 = loss_cls(predicted_class, target)
        loss2 = loss_recon(predicted_depth_map, depth)
        acc1, acc = sl_utils.accuracy(predicted_class, target, args, topk=(1, 5))

        # loss = init_cce * loss_annealing[anneal_step + i] * loss1 + args.loss_beta* loss2
        loss = 0.0 * loss1 + args.loss_beta * loss2
        loss.backward()

        if utils.is_primary(args) and args.save_depth:
            depth *= 255
            depth = depth.detach().cpu().numpy().astype(np.uint8)
            predicted_depth_map *= 255
            predicted_depth_map = (
                predicted_depth_map.detach().cpu().numpy().astype(np.uint8)
            )

            depth_target_pred_image = cv2.hconcat(
                [depth[0][0], predicted_depth_map[0][0]]
            )
            cv2.imwrite(
                os.path.join(args.train_depth_dir, str(i) + ".png"),
                depth_target_pred_image,
            )

        metric_logger.update(mse=loss2.item())
        metric_logger.update(cce=loss1.item())
        # metric_logger.update(an_cce = loss_annealing[anneal_step + i]*loss1.item())
        metric_logger.update(scaled_mse=args.loss_beta * loss2.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(top1_accuracy=acc1.item())
        metric_logger.update(top5_accuracy=acc.item())

        if len(lr_scheduler_values) > 0:
            for k, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_scheduler_values[
                    start_step + i
                ]  # * param_group['lr_scale']

        optimizer.step()
        metric_logger.synchronize_between_processes()

        if (
            utils.is_primary(args)
            and log_writer != None
            and ((start_step + i) % args.log_interval == 0)
        ):

            log_writer.set_step(start_step + i)
            log_writer.update(train_cce=metric_logger.cce.avg, head="train")
            log_writer.update(train_mse=metric_logger.mse.avg, head="train")
            log_writer.update(train_loss=metric_logger.loss.avg, head="train")
            log_writer.update(
                train_top1_accuracy=metric_logger.top1_accuracy.avg, head="train"
            )
            log_writer.update(
                train_top5_accuracy=metric_logger.top5_accuracy.avg, head="train"
            )
            log_writer.update(
                train_scaled_mse=metric_logger.scaled_mse.avg, head="train"
            )
            # log_writer.update(train_ann_cce = metric_logger.an_cce.avg, head = 'train')
            log_writer.update(epoch=start_epoch, head="train")
            # log_writer.update(loss_annealing = loss_annealing[anneal_step + i], head = 'train')
            log_writer.update(
                commit=True,
                learning_rate=lr_scheduler_values[start_step + i],
                head="train",
            )
            # annealing_count = (start_step+i)//args.log_interval

    return OrderedDict(
        [
            ("loss", metric_logger.loss.global_avg),
            ("top1", metric_logger.top1_accuracy.global_avg),
            ("top5", metric_logger.top5_accuracy.global_avg),
        ]
    )


def validate(
    model,
    start_epoch,
    val_dataloader,
    loss_fn,
    loss_recon,
    device,
    log_writer=None,
    start_step=0,
):

    model.eval()
    metric_logger = sl_utils.MetricLogger(delimiter="  ")
    header = "EVAL epoch: [{}]".format(start_epoch)

    if args.save_depth:
        if args.save_ddir:
            os.makedirs(args.save_ddir, exist_ok=True)
        else:
            args.save_depth = False

    for i, (input, depth, target) in enumerate(tqdm(val_dataloader)):
        input, depth, target = input.to(device), depth.to(device), target.to(device)
        output = model(input)
        if isinstance(output, (tuple, list)):
            predicted_depth_map, predicted_class = output

        loss1 = loss_fn(predicted_class, target)
        loss2 = loss_recon(predicted_depth_map, depth)

        loss = args.loss_alpha * loss1 + (args.loss_beta * loss2)

        acc1, acc = sl_utils.accuracy(predicted_class, target, args, topk=(1, 5))

        metric_logger.update(mse=loss2.item())
        metric_logger.update(cce=loss1.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(scaled_mse=args.loss_beta * loss2.item())
        metric_logger.update(top1_accuracy=acc1.item())
        metric_logger.update(top5_accuracy=acc.item())
        metric_logger.synchronize_between_processes()

        if utils.is_primary(args) and args.save_depth:

            depth *= 255
            depth = depth.detach().cpu().numpy().astype(np.uint8)
            predicted_depth_map *= 255
            predicted_depth_map = (
                predicted_depth_map.detach().cpu().numpy().astype(np.uint8)
            )

            depth_target_pred_image = cv2.hconcat(
                [depth[0][0], predicted_depth_map[0][0]]
            )
            cv2.imwrite(
                os.path.join(args.save_depth_dir, str(i) + ".png"),
                depth_target_pred_image,
            )

    if utils.is_primary(args) and log_writer != None:
        log_writer.set_step(start_step)
        log_writer.update(val_cce=metric_logger.cce.global_avg, head="val")
        log_writer.update(val_mse=metric_logger.mse.global_avg, head="val")
        log_writer.update(val_loss=metric_logger.loss.global_avg, head="val")
        log_writer.update(
            val_top1_accuracy=metric_logger.top1_accuracy.global_avg, head="val"
        )
        log_writer.update(
            val_top5_accuracy=metric_logger.top5_accuracy.global_avg, heaad="val"
        )
        log_writer.update(
            train_scaled_mse=metric_logger.scaled_mse.global_avg, head="val"
        )
        log_writer.update(commit=True, epoch=start_epoch, head="val")

    return OrderedDict(
        [
            ("loss", metric_logger.loss.global_avg),
            ("top1", metric_logger.top1_accuracy.global_avg),
            ("top5", metric_logger.top5_accuracy.global_avg),
        ]
    )


# main function
def main():
    num_epochs = args.epochs
    log_writer = None
    device = utils.init_distributed_device(args)

    args.save_depth_dir = os.path.join(args.save_ddir, "val_depth")
    args.train_depth_dir = os.path.join(args.save_ddir, "train_depth")

    if utils.is_primary(args):
        os.makedirs(args.save_depth_dir, exist_ok=True)
        os.makedirs(args.train_depth_dir, exist_ok=True)

        print(f"Is distributed training : {args.distributed}")
        if args.distributed:
            _logger.info(
                "Training in distributed mode with multiple processes, 1 device per process."
                f"Process {args.rank}, total {args.world_size}, device {args.device}."
            )
        else:
            _logger.info(f"Training with a single process on 1 device ({args.device}).")

        if args.log_wandb and args.log_dir != None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = WandBLogger(log_dir=args.log_dir, args=args)
        else:
            log_writer = None

    assert args.rank >= 0

    image_transformation = get_image_transform()
    depth_transformation = get_depth_transform()
    train_depth_dir = os.path.join(args.depth_dir, "train")
    dataset_train = LOCDataset(
        args.train_data_dir,
        train_depth_dir,
        image_transform=image_transformation,
        depth_transform=depth_transformation,
    )
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True
    )

    val_depth_dir = os.path.join(args.depth_dir, "val")
    dataset_eval = LOCDataset(
        args.val_data_dir,
        val_depth_dir,
        image_transform=image_transformation,
        depth_transform=depth_transformation,
    )
    sampler_eval = torch.utils.data.DistributedSampler(
        dataset_eval, num_replicas=args.world_size, rank=args.rank, shuffle=False
    )

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        checkpoint_path=args.initial_checkpoint,
    )
    model = model.to(device)

    if utils.is_primary(args):
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    if args.distributed:
        if utils.is_primary(args):
            _logger.info("Using native Torch DistributedDataParallel.")
        model = NativeDDP(model, device_ids=[device], find_unused_parameters=False)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size
    )
    loader_eval = torch.utils.data.DataLoader(
        dataset_eval, sampler=sampler_eval, batch_size=args.batch_size
    )

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
        train_cls_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(
            device
        )
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
            model=model,
            optimizer=optimizer,
            args=args,
            checkpoint_dir=args.output,
            recovery_dir=args.output,
            decreasing=eval_metrics,
            max_history=args.checkpoint_hist,
        )

    num_training_steps_per_epoch = len(loader_train)
    updates_per_epoch = len(loader_train)
    annealing_steps = num_training_steps_per_epoch * (args.epochs - args.rest_cce) + 1

    annealing_values = sl_utils.frange_cycle_sigmoid(
        0.0, 1.0, annealing_steps, n_cycle=4, ratio=1.0
    )

    lr_scheduler = None
    args.lr = args.warmup_lr * args.world_size

    lr_scheduler_values = sl_utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
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

    if utils.is_primary(args):
        print("Parameters used")
        print(f"Warmup Lr - {args.warmup_lr}")
        print(f"Base LR - {args.lr}")
        print(f"weight decay - {args.weight_decay}")
        print(f"Batch Size - {args.batch_size}")
        print(f"Epochs - {args.epochs}")
        print(f"Warmup epochs - {args.warmup_epochs}")
        print(f"Loss alphs - {args.loss_alpha}")
        print(f"Label Smoothing - {args.smoothing}")
        print(f"Total Annealing steps - {annealing_steps}")
        print(f"cce rest : {args.rest_cce}")

    for epoch in range(start_epoch, num_epochs):
        if hasattr(dataset_train, "set_epoch"):
            dataset_train.set_epoch(epoch)
        elif args.distributed and hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        if utils.is_primary(args) and log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model,
            epoch,
            loader_train,
            train_cls_loss_fn,
            train_recon_loss_fn,
            optimizer,
            args.rest_cce,
            device,
            lr_scheduler_values,
            annealing_values,
            lr_scheduler,
            log_writer,
        )
        val_stats = validate(
            model,
            epoch,
            loader_eval,
            validate_cls_loss_fn,
            validate_recon_loss_fn,
            device,
            log_writer,
            ((epoch + 1) * num_training_steps_per_epoch) + 1,
        )

        if utils.is_primary(args):
            print(
                f"Training::::: Loss={train_stats['loss']}, Top-1 Accuracy={train_stats['top1']}, Top-5 Accuracy={train_stats['top5']}"
            )
            print(
                f"Validation::: Loss={val_stats['loss']}, Top-1 Accuracy={val_stats['top1']}, Top-5 Accuracy={val_stats['top5']}"
            )

        if utils.is_primary(args):
            if saver is not None:
                best_metric, best_epoch = saver.save_checkpoint(
                    epoch, metric=val_stats["loss"]
                )

            if log_writer is not None:
                log_writer.flush()

    if utils.is_primary(args):
        print(best_metric, best_epoch)


if __name__ == "__main__":
    main()
