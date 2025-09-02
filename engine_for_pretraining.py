import gc
import math
import sys
import numpy as np
from typing import Iterable
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import utils
from utils import gather_predictions, gather_predictions_nontensor


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    # save grad norms
    if_dist = dist.is_initialized()
    qkv_grad_norms = np.zeros(shape=(12, 6, 5), dtype=np.float64)
    proj_grad_norms = np.zeros(shape=(12, 6), dtype=np.float64)
    patch_embed_grad_norms = np.zeros(shape=(2,), dtype=np.float64)

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        gc.collect()
        torch.cuda.empty_cache()
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs = model(videos, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        grad_norms = utils.collect_grad_norms_pretrain(model, num_layers=12, num_heads=6)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        if grad_norms is not None:
            qkv_grad_norms_iter, proj_grad_norms_iter, patch_embed_grad_norms_iter = grad_norms
            qkv_grad_norms += qkv_grad_norms_iter
            proj_grad_norms += proj_grad_norms_iter
            patch_embed_grad_norms += patch_embed_grad_norms_iter

        oshape = outputs.shape[0]
        del loss
        del videos
        del bool_masked_pos
        del outputs
        torch.cuda.synchronize()

        if step % (print_freq*5) == 0:
            # Print memory usage after each iteration
            print(f"AFTER Batch: {step}, total batch size {oshape}")
            utils.print_memory_usage()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    
    if if_dist:
        qkv_grad_norms = gather_predictions_nontensor(qkv_grad_norms, world_size=dist.get_world_size())
        proj_grad_norms = gather_predictions_nontensor(proj_grad_norms, world_size=dist.get_world_size())
        patch_embed_grad_norms = gather_predictions_nontensor(patch_embed_grad_norms, world_size=dist.get_world_size())
        qkv_grad_norms = np.sum(qkv_grad_norms, axis=0)
        proj_grad_norms = np.sum(proj_grad_norms, axis=0)
        patch_embed_grad_norms = np.sum(patch_embed_grad_norms, axis=0)

    assert np.max(qkv_grad_norms) > 0., "Point 1"
    qkv_grad_norms = qkv_grad_norms / len(data_loader)
    assert np.max(qkv_grad_norms) > 0., "Point 2"
    proj_grad_norms = proj_grad_norms / len(data_loader)
    patch_embed_grad_norms = patch_embed_grad_norms / len(data_loader)
    grad_norms = {"qkv": qkv_grad_norms.copy(), "proj": proj_grad_norms.copy(), "patch_embed": patch_embed_grad_norms.copy()}
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, grad_norms


def train_one_epoch_double(model: torch.nn.Module, data_loader1: Iterable, data_loader2: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, tubelet_size=2, with_grad_norms=False):
    model.train()
    print("Model is in train mode")
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    # save grad norms
    if_dist = dist.is_initialized()
    qkv_grad_norms = np.zeros(shape=(len(model.module.encoder.blocks), model.module.encoder.num_heads, 5), dtype=np.float64)
    proj_grad_norms = np.zeros(shape=(len(model.module.encoder.blocks), 6), dtype=np.float64)
    patch_embed_grad_norms = np.zeros(shape=(2,), dtype=np.float64)

    #for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    # Iterate over both DataLoaders simultaneously
    for step, (batch1, batch2) in enumerate(metric_logger.log_every(
        zip(data_loader1, data_loader2), 
        print_freq, 
        header,
        len(data_loader1)
        )):
        gc.collect()
        torch.cuda.empty_cache()

        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # Concatenate batches from both datasets
        videos1, mask1 = batch1
        videos2, mask2 = batch2
        videos = torch.cat([videos1, videos2], dim=0).to(device, non_blocking=True)
        bool_masked_pos = torch.cat([mask1, mask2], dim=0).to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=tubelet_size, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=tubelet_size, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs = model(videos, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        #grad_norms = utils.collect_grad_norms_pretrain(model, num_layers=12, num_heads=6)
        grad_norms = None
        loss_scale_value = loss_scaler.state_dict()["scale"]

        if grad_norms is not None:
            qkv_grad_norms_iter, proj_grad_norms_iter, patch_embed_grad_norms_iter = grad_norms
            qkv_grad_norms += qkv_grad_norms_iter
            proj_grad_norms += proj_grad_norms_iter
            patch_embed_grad_norms += patch_embed_grad_norms_iter

        oshape = outputs.shape[0]
        del loss
        del videos
        del bool_masked_pos
        del outputs
        torch.cuda.synchronize()

        if step % (print_freq*5) == 0:
            # Print memory usage after each iteration
            print(f"AFTER Batch: {step}, total batch size {oshape}")
            utils.print_memory_usage()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    if with_grad_norms:
        if if_dist:
            qkv_grad_norms = gather_predictions_nontensor(qkv_grad_norms, world_size=dist.get_world_size())
            proj_grad_norms = gather_predictions_nontensor(proj_grad_norms, world_size=dist.get_world_size())
            patch_embed_grad_norms = gather_predictions_nontensor(patch_embed_grad_norms, world_size=dist.get_world_size())
            qkv_grad_norms = np.sum(qkv_grad_norms, axis=0)
            proj_grad_norms = np.sum(proj_grad_norms, axis=0)
            patch_embed_grad_norms = np.sum(patch_embed_grad_norms, axis=0)

        # out epoch len is len(data_loader1)=len(data_loader2)
        assert np.max(qkv_grad_norms) > 0., "Point 1"
        qkv_grad_norms = qkv_grad_norms / len(data_loader1)
        assert np.max(qkv_grad_norms) > 0., "Point 2"
        proj_grad_norms = proj_grad_norms / len(data_loader1)
        patch_embed_grad_norms = patch_embed_grad_norms / len(data_loader1)
        grad_norms = {"qkv": qkv_grad_norms.copy(), "proj": proj_grad_norms.copy(), "patch_embed": patch_embed_grad_norms.copy()}
    else: 
        grad_norms = None

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, grad_norms


def train_one_epoch_triple(model: torch.nn.Module, data_loader1: Iterable, data_loader2: Iterable, data_loader3: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    # save grad norms
    if_dist = dist.is_initialized()
    qkv_grad_norms = np.zeros(shape=(12, 6, 5), dtype=np.float64)
    proj_grad_norms = np.zeros(shape=(12, 6), dtype=np.float64)
    patch_embed_grad_norms = np.zeros(shape=(2,), dtype=np.float64)

    #for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    # Iterate over both DataLoaders simultaneously
    for step, (batch1, batch2, batch3) in enumerate(metric_logger.log_every(
        zip(data_loader1, data_loader2, data_loader3), 
        print_freq, 
        header,
        len(data_loader1)
        )):
        gc.collect()
        torch.cuda.empty_cache()
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # Concatenate batches from both datasets
        videos1, mask1 = batch1
        videos2, mask2 = batch2
        videos3, mask3 = batch2
        videos = torch.cat([videos1, videos2, videos3], dim=0).to(device, non_blocking=True)
        bool_masked_pos = torch.cat([mask1, mask2, mask3], dim=0).to(device, non_blocking=True).flatten(1).to(torch.bool)


        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs = model(videos, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        grad_norms = utils.collect_grad_norms_pretrain(model, num_layers=12, num_heads=6)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        if grad_norms is not None:
            qkv_grad_norms_iter, proj_grad_norms_iter, patch_embed_grad_norms_iter = grad_norms
            qkv_grad_norms += qkv_grad_norms_iter
            proj_grad_norms += proj_grad_norms_iter
            patch_embed_grad_norms += patch_embed_grad_norms_iter

        oshape = outputs.shape[0]
        del loss
        del videos
        del bool_masked_pos
        del outputs
        torch.cuda.synchronize()

        if step % (print_freq*5) == 0:
            # Print memory usage after each iteration
            print(f"AFTER Batch: {step}, total batch size {oshape}")
            utils.print_memory_usage()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    if if_dist:
        qkv_grad_norms = gather_predictions_nontensor(qkv_grad_norms, world_size=dist.get_world_size())
        proj_grad_norms = gather_predictions_nontensor(proj_grad_norms, world_size=dist.get_world_size())
        patch_embed_grad_norms = gather_predictions_nontensor(patch_embed_grad_norms, world_size=dist.get_world_size())
        qkv_grad_norms = np.sum(qkv_grad_norms, axis=0)
        proj_grad_norms = np.sum(proj_grad_norms, axis=0)
        patch_embed_grad_norms = np.sum(patch_embed_grad_norms, axis=0)

    # out epoch len is len(data_loader1)=len(data_loader2)
    assert np.max(qkv_grad_norms) > 0., "Point 1"
    qkv_grad_norms = qkv_grad_norms / len(data_loader1)
    assert np.max(qkv_grad_norms) > 0., "Point 2"
    proj_grad_norms = proj_grad_norms / len(data_loader1)
    patch_embed_grad_norms = patch_embed_grad_norms / len(data_loader1)
    grad_norms = {"qkv": qkv_grad_norms.copy(), "proj": proj_grad_norms.copy(), "patch_embed": patch_embed_grad_norms.copy()}
    

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, grad_norms
