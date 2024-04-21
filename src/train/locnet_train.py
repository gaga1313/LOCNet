import logging
import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import random
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.loss import LabelSmoothingCrossEntropy
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs

from src.data import LOCDataset, get_shared_transform, get_img_transform
from src.sl_utils import WandBLogger
from src import sl_utils
from src.models import *
from src.utils.args import args

_logger = logging.getLogger("train")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


# train for one epoch
def train_one_epoch(
        model,
        epoch,
        train_dataloader,
        loss_cls,
        loss_recon,
        optimizer,
        rest_cce,
        device,
        lr_scheduler_values=[],
        loss_annealing=[],
        log_writer=None,
):
    model.train()

    metric_logger = sl_utils.MetricLogger(delimiter=" ")
    start_step = epoch * len(train_dataloader) + 1

    anneal_step = (epoch - rest_cce) * len(train_dataloader)
    if anneal_step < 0:
        anneal_step = 0

    if epoch >= rest_cce:
        loss_alpha = args.loss_alpha
    else:
        loss_alpha = 0.0

    for i, (input, depth, target) in enumerate(tqdm(train_dataloader)):
        input, depth, target = input.to(device), depth.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(input)

        predicted_depth_map, predicted_class = output

        loss1 = loss_cls(predicted_class, target)
        loss2 = loss_recon(predicted_depth_map, depth)

        # loss = (
        #         args.loss_alpha * loss_annealing[anneal_step + i] * loss1
        #         + args.loss_beta * (1 - loss_annealing[anneal_step + i]) * loss2
        # )
        loss = loss_alpha * loss1 + args.loss_beta * loss2
        loss.backward()

        acc1, acc5 = sl_utils.accuracy(predicted_class, target, topk=(1, 5))

        if utils.is_primary(args) and args.save_ddir != None:
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
        metric_logger.update(
            scaled_cce=loss_alpha * loss_annealing[anneal_step + i] * loss1.item()
        )
        metric_logger.update(scaled_mse=args.loss_beta * loss2.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(top1_accuracy=acc1.item())
        metric_logger.update(top5_accuracy=acc5.item())

        if len(lr_scheduler_values) > 0:
            for k, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_scheduler_values[start_step + i]

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
                learning_rate=lr_scheduler_values[start_step + i],
                head="train",
            )
            log_writer.update(
                train_scaled_mse=metric_logger.scaled_mse.avg, head="train"
            )
            log_writer.update(
                train_scale_cce=metric_logger.scaled_cce.avg, head="train"
            )
            log_writer.update(epoch=epoch, head="train")
            log_writer.update(
                commit=True,
                loss_annealing=loss_annealing[anneal_step + i],
                head="train",
            )

    return OrderedDict(
        [
            ("loss", metric_logger.loss.avg),
            ("top1", metric_logger.top1_accuracy.avg),
            ("top5", metric_logger.top5_accuracy.avg),
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

    if args.save_ddir != None:
        os.makedirs(args.save_ddir, exist_ok=True)

    with torch.no_grad():
        for i, (input, depth, target) in enumerate(tqdm(val_dataloader)):
            input, depth, target = input.to(device), depth.to(device), target.to(device)

            output = model(input)
            predicted_depth_map, predicted_class = output

            loss1 = loss_fn(predicted_class, target)
            loss2 = loss_recon(predicted_depth_map, depth)
            loss = args.loss_alpha * loss1 + (args.loss_beta * loss2)

            acc1, acc5 = sl_utils.accuracy(predicted_class, target, topk=(1, 5))

            metric_logger.update(mse=loss2.item())
            metric_logger.update(cce=loss1.item())
            metric_logger.update(loss=loss.item())

            metric_logger.update(top1_accuracy=acc1.item())
            metric_logger.update(top5_accuracy=acc5.item())
            metric_logger.synchronize_between_processes()

            if utils.is_primary(args) and args.save_ddir != None:
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

    train_shared_transformations = get_shared_transform("train")
    train_image_transformations = get_img_transform("train")

    train_depth_dir = os.path.join(args.depth_dir, "train")
    train_image_dir = os.path.join(args.image_dir, "train")

    dataset_train = LOCDataset(
        train_image_dir,
        train_depth_dir,
        shared_transforms=train_shared_transformations,
        img_transform=train_image_transformations,
    )
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True
    )

    val_depth_dir = os.path.join(args.depth_dir, "val")
    val_image_dir = os.path.join(args.image_dir, "val")
    val_shared_transformations = get_shared_transform("val")
    val_image_transformations = get_img_transform("val")

    dataset_eval = LOCDataset(
        val_image_dir,
        val_depth_dir,
        shared_transforms=val_shared_transformations,
        img_transform=val_image_transformations,
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
    train_recon_loss_fn = nn.MSELoss().to(device)  # reconstruction Loss

    validate_cls_loss_fn = nn.CrossEntropyLoss().to(device)
    validate_recon_loss_fn = nn.MSELoss().to(device)

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
    n_cycle = 4

    # import ipdb;ipdb.set_trace()
    annealing_steps = num_training_steps_per_epoch * (args.epochs - args.rest_cce) + 1
    # annealing_values = sl_utils.frange_cycle_sigmoid(
    #     1.0, 1.0, annealing_steps, n_cycle=1, ratio=1.0
    # )
    mask = np.array([1 for i in range(annealing_steps)]) #changed
    # steps_per_cycle = annealing_steps//n_cycle
    # for i in range(1,n_cycle,2):
    #     mask[i*steps_per_cycle : (i+1)*steps_per_cycle] = 1.0
    annealing_values =  mask
    
    # args.mse_scale = 
    # args.lr = args.warmup_lr * args.world_size

    lr_scheduler_values = sl_utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )
    n = len(lr_scheduler_values)
    lr_scheduler_values = [args.lr for i in range(n)]

    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch

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
            ((epoch + 1) * num_training_steps_per_epoch),
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
