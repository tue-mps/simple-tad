import os
import gc
import cv2
import numpy as np
import pandas as pd
import math
import sys
from typing import Iterable, Optional
import torch
from mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
import torchmetrics
from sklearn.metrics import auc
import torch.distributed as dist
import matplotlib.pyplot as plt
import seaborn as sns

from dataset.vis_tools import threshold_curve_plots
from utils import gather_predictions, gather_predictions_nontensor
from anaysis.metrics import calculate_MORE_metrics, THRESHOLDS

#THRESHOLDS = np.arange(0.00, 1.001, 0.01).tolist()


def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


def train_class_batch_ttc(model, samples, target, ttc, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target, ttc)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, with_ttc=False, smoothed_labels_for_loss=False, 
                    get_grad_norms=True):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if hasattr(data_loader, 'batch_sampler'):
        batch_size = data_loader.batch_sampler.batch_size
    elif hasattr(data_loader, 'batch_size'):
        batch_size = data_loader.batch_size
    else:
        raise ValueError("Unable to determine batch size from DataLoader.")

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    preds = []
    labels = []
    if_dist = dist.is_initialized()

    # save grad norms
    if get_grad_norms:
        qkv_grad_norms = np.zeros(shape=(len(model.blocks), model.num_heads, 5), dtype=np.float64)
        proj_grad_norms = np.zeros(shape=(len(model.blocks), 6), dtype=np.float64)
        patch_embed_grad_norms = np.zeros(shape=(2,), dtype=np.float64)
    else:
        qkv_grad_norms = None
        proj_grad_norms = None
        patch_embed_grad_norms = None

    for data_iter_step, (samples, targets, _, extra_info) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        gc.collect()  # Run garbage collection to clear unused memory
        torch.cuda.empty_cache()  # Free up CUDA memory

        #print(f"INPUT SHAPE: {samples.shape}")

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        #print(f"Targets: {targets}")

        # collect labels
        labels.append(targets.detach().to(device, non_blocking=True) if if_dist else targets.detach().cpu())  # !!
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # print("extra info")
        # print("type", type(extra_info))
        # print(extra_info)
        # exit(0)
        if smoothed_labels_for_loss:
            targets_loss = extra_info["smoothed_labels"].to(device, non_blocking=True)
        else:
            targets_loss = targets
        if with_ttc:
            ttc = extra_info["ttc"]
            ttc = ttc.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            if with_ttc:
                loss, output = train_class_batch_ttc(
                    model, samples, targets_loss, ttc, criterion)
            else:
                loss, output = train_class_batch(
                    model, samples, targets_loss, criterion)
            # collect predictions
            preds.append(output.detach() if if_dist else output.detach().cpu())   # !!
        else:
            with torch.cuda.amp.autocast():
                if with_ttc:
                    loss, output = train_class_batch_ttc(
                        model, samples, targets_loss, ttc, criterion)
                else:
                    loss, output = train_class_batch(
                        model, samples, targets_loss, criterion)
            # collect predictions
            preds.append(output.detach() if if_dist else output.detach().cpu())    # !!

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            if get_grad_norms:
                grad_norms = utils.collect_grad_norms(model, num_layers=12, num_heads=6) if get_grad_norms else None
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if get_grad_norms:
                grad_norms = utils.collect_grad_norms(model, num_layers=len(model.blocks), num_heads=model.num_heads) if get_grad_norms else None
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        if get_grad_norms and grad_norms is not None:
            qkv_grad_norms_iter, proj_grad_norms_iter, patch_embed_grad_norms_iter = grad_norms
            qkv_grad_norms += qkv_grad_norms_iter
            proj_grad_norms += proj_grad_norms_iter
            patch_embed_grad_norms += patch_embed_grad_norms_iter

        del loss
        del samples
        torch.cuda.synchronize()

        if data_iter_step % print_freq == 0:
            # Print memory usage after each iteration
            print(f"AFTER Batch: {data_iter_step}, total batch size {output.shape[0]}")
            utils.print_memory_usage()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        del output
        del targets
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
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
            log_writer.update(loss=loss_value, head="train")
            log_writer.update(class_acc=class_acc, head="train")
            log_writer.update(loss_scale=loss_scale_value, head="train_opt")
            log_writer.update(lr=max_lr, head="train_opt")
            log_writer.update(min_lr=min_lr, head="train_opt")
            log_writer.update(weight_decay=weight_decay_value, head="train_opt")
            log_writer.update(grad_norm=grad_norm, head="train_opt")

            log_writer.set_step()

    # Calculate total metrics
    if if_dist and get_grad_norms:
        preds = gather_predictions_nontensor(preds, world_size=dist.get_world_size())
        labels = gather_predictions_nontensor(labels, world_size=dist.get_world_size())
        qkv_grad_norms = gather_predictions_nontensor(qkv_grad_norms, world_size=dist.get_world_size())
        proj_grad_norms = gather_predictions_nontensor(proj_grad_norms, world_size=dist.get_world_size())
        patch_embed_grad_norms = gather_predictions_nontensor(patch_embed_grad_norms, world_size=dist.get_world_size())
        qkv_grad_norms = np.sum(qkv_grad_norms, axis=0)
        proj_grad_norms = np.sum(proj_grad_norms, axis=0)
        patch_embed_grad_norms = np.sum(patch_embed_grad_norms, axis=0)
    all_preds = torch.cat(preds, dim=0)
    all_labels = torch.cat(labels, dim=0)
    if get_grad_norms:
        assert np.max(qkv_grad_norms) > 0., "Point 1"
        qkv_grad_norms = qkv_grad_norms / len(data_loader)
        assert np.max(qkv_grad_norms) > 0., "Point 2"
        proj_grad_norms = proj_grad_norms / len(data_loader)
        patch_embed_grad_norms = patch_embed_grad_norms / len(data_loader)
        grad_norms = {"qkv": qkv_grad_norms.copy(), "proj": proj_grad_norms.copy(), "patch_embed": patch_embed_grad_norms.copy()}
    else:
        grad_norms = None
    my_metrics = {}
    metr_acc, recall, precision, f1, confmat, auroc, ap, pr_curve, roc_curve, _ = calculate_metrics(all_preds, all_labels)
    # Log them
    my_metrics['metr_acc'] = metr_acc
    my_metrics['recall'] = recall
    my_metrics['precision'] = precision
    my_metrics['f1'] = f1
    my_metrics['auroc'] = auroc
    my_metrics['ap'] = ap
    # extra metrics
    values = torch.nn.functional.softmax(all_preds, dim=1)
    values = values[:, 1]
    my_metrics['logitsP_mean'] = all_preds[:, 1].mean().item()
    my_metrics['logitsP_std'] = all_preds[:, 1].std().item()
    my_metrics['logitsP_median'] = all_preds[:, 1].median().item()
    my_metrics['logitsN_mean'] = all_preds[:, 0].mean().item()
    my_metrics['logitsN_std'] = all_preds[:, 0].std().item()
    my_metrics['logitsN_median'] = all_preds[:, 0].median().item()
    my_metrics['probs_mean'] = values.mean().item()
    my_metrics['probs_std'] = values.std().item()
    my_metrics['probs_median'] = values.median().item()
    # plot figures
    plots = plot_figures(confmat, pr_curve, roc_curve)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, my_metrics, plots, grad_norms


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, with_ttc=False, smoothed_labels_for_loss=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    preds = []
    labels = []

    i = 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        i += 1
        gc.collect()  # Run garbage collection to clear unused memory
        torch.cuda.empty_cache()  # Free up CUDA memory
        videos = batch[0]
        target = batch[1]
        ttc = batch[3]["ttc"] if with_ttc else None
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if with_ttc:
            ttc = ttc.to(device, non_blocking=True)
        if smoothed_labels_for_loss:
            targets_loss = batch[3]["smoothed_labels"].to(device, non_blocking=True)
        else:
            targets_loss = target

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            if with_ttc:
                loss = criterion(output, targets_loss, ttc)
            else:
                loss = criterion(output, targets_loss)

        # collect predictions
        preds.append(output.detach() if dist.is_initialized() else output.cpu().detach())
        labels.append(target.detach().to(device, non_blocking=True) if dist.is_initialized() else target.cpu().detach())

        acc = accuracy(output, target, topk=(1,))[0]

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)

        osh = output.shape[0]
        del output
        del target
        del loss

        # Print memory usage after some iterations
        if i % 30 == 0:
            print(f"AFTER Batch: {i}, total batch size {osh}")
            utils.print_memory_usage()


    # Calculate total metrics
    if dist.is_initialized():
        preds = gather_predictions_nontensor(preds, world_size=dist.get_world_size())
        labels = gather_predictions_nontensor(labels, world_size=dist.get_world_size())
    all_preds = torch.cat(preds, dim=0)
    all_labels = torch.cat(labels, dim=0)
    metr_acc, recall, precision, f1, confmat, auroc, ap, pr_curve, roc_curve, mcc_metrics = calculate_metrics(all_preds, all_labels)
    mcc_auc, mcc_max, mcc_max_threshold, mcc_05 = mcc_metrics
    # Log them
    my_metrics = {}
    my_metrics['metr_acc'] = metr_acc
    my_metrics['recall'] = recall
    my_metrics['precision'] = precision
    my_metrics['f1'] = f1
    my_metrics['auroc'] = auroc
    my_metrics['ap'] = ap
    my_metrics['mcc_auc'] = mcc_auc
    my_metrics['mcc_max'] = mcc_max
    my_metrics['mcc_max_thresh'] = mcc_max_threshold
    my_metrics['mcc_05'] = mcc_05
    # extra metrics
    values = torch.nn.functional.softmax(all_preds, dim=1)
    values = values[:, 1]
    my_metrics['logitsP_mean'] = all_preds[:, 1].mean().item()
    my_metrics['logitsP_std'] = all_preds[:, 1].std().item()
    my_metrics['logitsP_median'] = all_preds[:, 1].median().item()
    my_metrics['logitsN_mean'] = all_preds[:, 0].mean().item()
    my_metrics['logitsN_std'] = all_preds[:, 0].std().item()
    my_metrics['logitsN_median'] = all_preds[:, 0].median().item()
    my_metrics['probs_mean'] = values.mean().item()
    my_metrics['probs_std'] = values.std().item()
    my_metrics['probs_median'] = values.median().item()
    #

    plots = plot_figures(confmat, pr_curve, roc_curve)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} AP {ap} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc, ap=ap, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, my_metrics, plots


@torch.no_grad()
def final_test(data_loader, model, device, preds_file, stats_file, plot_dir=None, with_ttc=False, smoothed_labels_for_loss=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    clips = []
    frame_names = []
    preds = []
    labels = []
    ttcs = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        extra_info = batch[3]

        ttc = extra_info["ttc"]
        clips_batch = extra_info["clip"]
        frame_batch = extra_info["frame"]

        clips.extend(clips_batch)
        frame_names.extend(frame_batch)
        labels.append(target.detach().to(device, non_blocking=True) if dist.is_initialized() else output.detach())
        ttcs.append(ttc.clone().detach().to(device, non_blocking=True) if dist.is_initialized() else ttc.clone().detach().cpu())

        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if with_ttc:
            ttc = ttc.to(device, non_blocking=True)

        if smoothed_labels_for_loss:
            targets_loss = extra_info["smoothed_labels"].to(device, non_blocking=True)
        else:
            targets_loss = target

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            if with_ttc:
                loss = criterion(output, targets_loss, ttc)
            else:
                loss = criterion(output, targets_loss)

        for i in range(output.size(0)):
            string = f"{ids[i]} {output.data[i].cpu().numpy().tolist()} {int(target[i].cpu().numpy())}\n"
            final_result.append(string)

        # collect predictions
        preds.append(output.detach() if dist.is_initialized() else output.cpu().detach())

        acc1 = accuracy(output, target, topk=(1,))[0]

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # calculate metrics
    if dist.is_initialized():
        preds = gather_predictions_nontensor(preds, world_size=dist.get_world_size())
        labels = gather_predictions_nontensor(labels, world_size=dist.get_world_size())
        ttcs = gather_predictions_nontensor(ttcs, world_size=dist.get_world_size())
        all_clips = gather_predictions_nontensor(clips, world_size=dist.get_world_size())
        all_filenames = gather_predictions_nontensor(frame_names, world_size=dist.get_world_size())
    else:
        all_clips = clips
        all_filenames = frame_names
    all_preds = torch.cat(preds, dim=0)
    all_labels = torch.cat(labels, dim=0)
    all_ttcs = torch.cat(ttcs, dim=0)
    logits = torch.clone(all_preds).detach()
    all_preds = torch.nn.functional.softmax(all_preds, dim=1)
    values = all_preds[:, 1]
    _, all_preds = torch.max(all_preds, 1)
    metr_acc = torchmetrics.functional.accuracy(preds=all_preds, target=all_labels, task="binary").item()
    recall = torchmetrics.functional.recall(preds=all_preds, target=all_labels, task="binary").item()
    precision = torchmetrics.functional.precision(preds=all_preds, target=all_labels, task="binary").item()
    f1 = torchmetrics.functional.f1_score(preds=all_preds, target=all_labels, task="binary").item()
    confmat = torchmetrics.functional.confusion_matrix(preds=all_preds, target=all_labels, task="binary").detach().tolist()
    auroc = torchmetrics.functional.auroc(
        preds=values,
        target=all_labels,
        task="binary",
        thresholds=THRESHOLDS
    ).item()
    ap = torchmetrics.functional.average_precision(
        preds=values,
        target=all_labels,
        task="binary",
        thresholds=THRESHOLDS
    ).item()
    pr_curve = torchmetrics.functional.precision_recall_curve(
        preds=values,
        target=all_labels,
        task="binary",
        thresholds=THRESHOLDS
    )
    roc_curve = torchmetrics.functional.roc(
        preds=values,
        target=all_labels,
        task="binary",
        thresholds=THRESHOLDS
    )
    lines = ["\n===================================",
             f"mAP: {ap}, auroc: {auroc}, acc: {metr_acc}",
             f"P@0.5: {precision}, R@0.5: {recall}, F1@0.5: {f1}",
             f"Confmat: \n\t{confmat[0][0]} | {confmat[0][1]} \n\t{confmat[1][0]} | {confmat[1][1]}",
             f"----------------------------"]
    with open(stats_file, "w") as f:
        for l in lines:
            f.write(l + "\n")
            print(l)
    
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        pr_precision, pr_recall, pr_thresholds = [item.detach().tolist() for item in pr_curve]
        roc_fpr, roc_tpr, roc_thresholds = [item.detach().tolist() for item in roc_curve]
        fig2 = threshold_curve_plots(
            x_values=pr_recall, y_values=pr_precision, thresholds=pr_thresholds + [1.],
            x_label="Recall", y_label="Precision", plot_name="PR curve",
            score=True, to_img=True, curve_type_correction="pr"
        )
        fig3 = threshold_curve_plots(
            x_values=roc_fpr, y_values=roc_tpr, thresholds=roc_thresholds,
            x_label="FP rate", y_label="TP rate", plot_name="ROC curve",
            score=True, to_img=True, curve_type_correction="roc"
        )
        cv2.imwrite(os.path.join(plot_dir, "pr.jpg"), fig2)
        cv2.imwrite(os.path.join(plot_dir, "roc.jpg"), fig3)

    # if not os.path.exists(file):
    #     os.mknod(file)
    # with open(file, 'w') as f:
    #     f.write("{}\n".format(acc1))
    #     for line in final_result:
    #         f.write(line)

    all_labels = all_labels.detach().cpu().numpy().astype(int)
    logits = logits.detach().cpu().numpy()
    all_ttcs = all_ttcs.detach().cpu().numpy()
    df = pd.DataFrame({
        "clip": all_clips,
        "filename": all_filenames,
        "logits_safe": logits[:, 0],
        "logits_risk": logits[:, 1],
        "label": all_labels,
        "ttc": all_ttcs
    })
    df.to_csv(preds_file, index=True, header=True)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    print("Reading individual output files")

    # TODO: combine two csv here
    for x in range(num_tasks):
        file = os.path.join(eval_path, f'predictions_{x}.csv')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=float, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
            dict_feats[name].append(data)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[2] for x in ans]
    final_top1 = np.mean(top1)
    return final_top1*100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    return [pred, top1, int(label)]


def calculate_metrics(preds, labels, do_softmax=True):
    if do_softmax:
        preds = torch.nn.functional.softmax(preds, dim=1)
    values = preds[:, 1]
    _, preds = torch.max(preds, 1)
    metr_acc = torchmetrics.functional.accuracy(preds=preds, target=labels, task="binary").item()
    recall = torchmetrics.functional.recall(preds=preds, target=labels, task="binary").item()
    precision = torchmetrics.functional.precision(preds=preds, target=labels, task="binary").item()
    f1 = torchmetrics.functional.f1_score(preds=preds, target=labels, task="binary").item()
    confmat = torchmetrics.functional.confusion_matrix(preds=preds, target=labels, task="binary").detach().tolist()
    auroc = torchmetrics.functional.auroc(
        preds=values,
        target=labels,
        task="binary",
        thresholds=THRESHOLDS
    ).item()
    ap = torchmetrics.functional.average_precision(
        preds=values,
        target=labels,
        task="binary",
        thresholds=THRESHOLDS
    ).item()
    pr_curve = torchmetrics.functional.precision_recall_curve(
        preds=values,
        target=labels,
        task="binary",
        thresholds=THRESHOLDS
    )
    roc_curve = torchmetrics.functional.roc(
        preds=values,
        target=labels,
        task="binary",
        thresholds=THRESHOLDS
    )
    # MCC
    thresh_metrics = calculate_MORE_metrics(preds=values.detach().cpu(), labels=labels.detach().cpu())
    mcc_thresholded_vals, p_thresholded_vals, r_thresholded_vals, acc_thresholded_vals, f1_thresholded_vals = thresh_metrics[
                                                                                                              -5:]
    mcc_max = max(mcc_thresholded_vals)
    mcc_max_idx = mcc_thresholded_vals.index(mcc_max)
    idx_05 = THRESHOLDS.index(0.5)
    mcc_05 = mcc_thresholded_vals[idx_05]
    mcc_auc = auc(THRESHOLDS, mcc_thresholded_vals)
    return metr_acc, recall, precision, f1, confmat, auroc, ap, pr_curve, roc_curve, (mcc_auc, mcc_max, THRESHOLDS[mcc_max_idx], mcc_05)


def plot_figures(confmat, pr_curve, roc_curve):
    # plot figures
    df_cm = pd.DataFrame(confmat)
    fig1 = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
    plt.close(fig1)
    pr_precision, pr_recall, pr_thresholds = pr_curve
    fig2 = threshold_curve_plots(
        x_values=pr_recall.tolist(), y_values=pr_precision.tolist(), thresholds=pr_thresholds.tolist() + [1.],
        x_label="Recall", y_label="Precision", plot_name="PR curve",
        score=True, to_img=False, curve_type_correction="pr"
    )
    plt.close(fig2)
    roc_fpr, roc_tpr, roc_thresholds = roc_curve
    fig3 = threshold_curve_plots(
        x_values=roc_fpr.tolist(), y_values=roc_tpr.tolist(), thresholds=roc_thresholds.tolist(),
        x_label="FP rate", y_label="TP rate", plot_name="ROC curve",
        score=True, to_img=False, curve_type_correction="roc"
    )
    plt.close(fig3)
    plots = {"confusion_matrix": fig1, "PR_curve": fig2, "ROC_curve": fig3}
    return plots