# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from utils import adjust_learning_rate


def multilabel_accuracy(output, targets, threshold=0.5):
    pred = torch.sigmoid(output) >= threshold
    truth = targets >= 0.5
    return pred.eq(truth).float().mean()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq
    use_amp = args.use_amp
    multilabel = getattr(args, 'multilabel', False)
    threshold = getattr(args, 'disfa_threshold', 0.5)
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if multilabel:
            targets = targets.float()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        
        torch.cuda.synchronize()

        if multilabel:
            au_acc = multilabel_accuracy(output, targets, threshold=threshold)
            class_acc = None
        elif mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        if multilabel:
            metric_logger.update(au_acc=au_acc)
        else:
            metric_logger.update(class_acc=class_acc)
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
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            if multilabel:
                log_writer.update(au_acc=au_acc, head="loss")
            else:
                log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(
    data_loader,
    model,
    device,
    use_amp=False,
    criterion=None,
    multilabel=False,
    threshold=0.5,
    au_labels=None,
    log_micro_f1=False,
):
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss() if multilabel else torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if multilabel:
        true_positives = None
        false_positives = None
        false_negatives = None

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if multilabel:
            target = target.float()

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                if isinstance(output, dict):
                    output = output['logits']
                loss = criterion(output, target)
        else:
            output = model(images)
            if isinstance(output, dict):
                output = output['logits']
            loss = criterion(output, target)

        torch.cuda.synchronize()

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        if multilabel:
            pred = torch.sigmoid(output) >= threshold
            truth = target >= 0.5
            batch_au_acc = pred.eq(truth).float().mean().item() * 100.0
            metric_logger.meters['au_acc'].update(batch_au_acc, n=batch_size)

            batch_tp = (pred & truth).sum(dim=0).to(dtype=torch.float32).cpu()
            batch_fp = (pred & ~truth).sum(dim=0).to(dtype=torch.float32).cpu()
            batch_fn = ((~pred) & truth).sum(dim=0).to(dtype=torch.float32).cpu()
            if true_positives is None:
                true_positives = batch_tp
                false_positives = batch_fp
                false_negatives = batch_fn
            else:
                true_positives += batch_tp
                false_positives += batch_fp
                false_negatives += batch_fn
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if multilabel:
        denom = (2 * true_positives + false_positives + false_negatives).clamp_min(1.0)
        per_au_f1 = ((2 * true_positives) / denom) * 100.0
        au_f1 = per_au_f1.mean().item()
        micro_denom = (2 * true_positives.sum() + false_positives.sum() + false_negatives.sum()).clamp_min(1.0)
        micro_f1 = ((2 * true_positives.sum()) / micro_denom).item() * 100.0
        if au_labels is None:
            au_labels = [f'au{i}' for i in range(len(per_au_f1))]
        summary = '* AU-Acc {au_acc.global_avg:.3f} AU-F1 {au_f1:.3f}'
        if log_micro_f1:
            summary += ' micro-F1 {micro_f1:.3f}'
        summary += ' loss {losses.global_avg:.3f}'
        print(summary.format(
            au_acc=metric_logger.au_acc,
            au_f1=au_f1,
            micro_f1=micro_f1,
            losses=metric_logger.loss,
        ))
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats['au_f1'] = au_f1
        stats['micro_f1'] = micro_f1
        stats['per_au_f1'] = {
            label: float(value) for label, value in zip(au_labels, per_au_f1.tolist())
        }
        return stats

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
