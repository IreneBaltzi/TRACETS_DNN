import math
import sys
from typing import Iterable
from utils import misc
from utils.losses import FocalLoss
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, balanced_accuracy_score
from utils.models import build_key_padding_mask
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.metrics_sepsis import evaluate_sepsis_from_lists


def weighted_masked_bce(
    logits: torch.Tensor,         # [B, T]
    y: torch.Tensor,              # [B, T] in {0,1}
    lengths: torch.Tensor,        # [B]
    pos_weight: float | None,    
    lam: float = 0.02,           
    beta_pos: float = 3.0,        
    favor: str = "early",         
    normalize: bool = True,      
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    
    B, T = y.shape
    pad = build_key_padding_mask(lengths, T)              # [B, T] True on PAD
    valid = ~pad

    
    t = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
    if favor == "late":
        dist = (lengths.unsqueeze(1) - 1 - t).clamp(min=0).float()
    else:  # "early"
        dist = t.float()

    
    w_time = torch.exp(-lam * dist)                        # [B, T]

   
    w_time = w_time * valid.float()

    if normalize:
        denom = w_time.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B,1]
        w_time = w_time * (lengths.clamp_min(1).unsqueeze(1).float() / denom)

    w = torch.where(y.bool(), w_time * beta_pos, w_time)   # [B, T]

    pw = None if pos_weight is None else torch.tensor([pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    elem = bce(logits, y.float())                          # [B, T]

    loss = (elem * w)[valid].mean()
    return loss

def _ss_prob(epoch: int, args) -> float:

    if not getattr(args, "ar_sched", None) or not getattr(args.ar_sched, "enable", False):
        return 0.0
    p_max = getattr(args.ar_sched, "p_max", 0.3)
    warm = max(1, getattr(args.ar_sched, "warmup_epochs", 5))
    p = p_max * min(1.0, max(0.0, epoch / float(warm)))
    return float(p)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, args=None):
    
    def shift_right_labels(y: torch.Tensor, pad_val: int=-1) -> torch.Tensor:
       
        B, T = y.shape
        y_prev = torch.empty_like(y)
        y_prev[:, 0] = pad_val
        y_prev[:, 1:] = y[:, :-1]
        return y_prev
    
    def masked_bce_with_logits(logits: torch.Tensor, y: torch.Tensor, lengths: torch.Tensor,
                               pos_weight: float | None, device: torch.device) -> torch.Tensor:
        pad = build_key_padding_mask(lengths, y.size(1))  # [B, T] True=PAD
        valid = ~pad
        pw = None if pos_weight is None else torch.tensor([pos_weight], device=device)
        bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
        return bce(logits[valid], y[valid].float()).mean(), valid.sum().item()

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if getattr(args, "ar_sched", None) and getattr(args.ar_sched, "enable", False):
        print(f"[sched-sampling] epoch={epoch} p_ss={_ss_prob(epoch, args):.3f}")
    print_freq = 10

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    for batch in metric_logger.log_every(data_loader, print_freq, header):   
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            samples, targets, lengths = batch
            lengths = lengths.to(device, non_blocking=True)
            is_sequence = True
        else:
            samples, targets = batch
            lengths = None
            is_sequence = False

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if not is_sequence and getattr(model, 'num_labels', 1) >= 2:
            targets = targets.squeeze().long()

        with torch.cuda.amp.autocast():
            if is_sequence:
                y_prev_gold = shift_right_labels(targets, pad_val=-1)
                logits_tf = model(samples, y_prev_gold, lengths)  # [B,T]

                pos_w = args.optim.bce.pos_weight if (args.optim.loss == 'bce' and getattr(args.optim.bce, "use_pos_weight", False)) else None
                use_time_weighted = bool(getattr(args.train, "use_time_weighted_loss", False))
                lam       = float(getattr(args.train, "time_decay_lambda", 0.02))
                beta_pos  = float(getattr(args.train, "time_pos_boost", 3.0))
                favor     = getattr(args.train, "time_weight_direction", "early")
                normalize = bool(getattr(args.train, "time_weight_normalize", True))

                if use_time_weighted:
                    loss_tf = weighted_masked_bce(
                        logits=logits_tf, y=targets, lengths=lengths,
                        pos_weight=pos_w, lam=lam, beta_pos=beta_pos,
                        favor=favor, normalize=normalize, device=device
                    )
                else:
                    loss_tf, _ = masked_bce_with_logits(logits_tf, targets, lengths, pos_w, device)

                p_ss = _ss_prob(epoch, args)  
                if p_ss > 0.0:
                    with torch.no_grad():
                        probs = torch.sigmoid(logits_tf)
                        preds = (probs > 0.5).long()  # [B,T]
                        y_prev_pred = torch.empty_like(targets)
                        y_prev_pred[:, 0]  = -1
                        y_prev_pred[:, 1:] = preds[:, :-1]

                    B, T = targets.shape
                    pad = build_key_padding_mask(lengths, T).to(device)  # True=PAD
                    valid = ~pad
                    rand = torch.rand_like(targets.float(), device=device)
                    mask_ss = (rand < p_ss) & valid
                    mask_ss[:, 0] = False 

                    y_prev_mix = y_prev_gold.clone()
                    y_prev_mix[mask_ss] = y_prev_pred[mask_ss]
                    logits_ss = model(samples, y_prev_mix, lengths)  # [B,T]

                    if use_time_weighted:
                        loss_ss = weighted_masked_bce(
                            logits=logits_ss, y=targets, lengths=lengths,
                            pos_weight=pos_w, lam=lam, beta_pos=beta_pos, device=device
                        )
                    else:
                        loss_ss, _ = masked_bce_with_logits(logits_ss, targets, lengths, pos_w, device)

                    loss = 0.5 * (loss_tf + loss_ss)
                    logits = logits_tf  
                else:
                    loss = loss_tf
                    logits = logits_tf

            else:
                logits = model(samples)
                loss = criterion(logits, targets)

        loss_value = float(loss.item())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), flush=True)
            sys.exit(1)

        optimizer.zero_grad()
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
def evaluate(data_loader, model, criterion, device, return_cm=False, features=None, args=None):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    if return_cm:
        cumulative_cm = None

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 5, header):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                sample, target, lengths = batch
                sample = sample.to(device, non_blocking=True)  # [B,T,F]
                target = target.to(device, non_blocking=True)  # [B,T]
                lengths = lengths.to(device, non_blocking=True)  # [B]

                threshold = getattr(args, 'threshold', 0.5)
                logits, y_hat = model.generate(sample, lengths, threshold=threshold, return_logits=True)

                if (y_hat == 1).any() and (target == 1).any():
                    rows_with_ones = (y_hat == 1).any(dim=1)
                    row_indices = torch.where(rows_with_ones)[0]
                    print("Rows in y_hat containing at least one 1:", row_indices)
                    print('Predictions:', y_hat[row_indices])
                    print('Targets:', target[row_indices])

                B, T = target.shape
                pad_mask = (torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1))  # [B, T]
                valid = ~pad_mask

                if isinstance(criterion, nn.BCEWithLogitsLoss) and getattr(criterion, 'reduction', 'mean') != 'none':
                    pos_w = getattr(criterion, 'pos_weight', None)
                    bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_w)
                    elem_loss = bce(logits[valid], target[valid].float())
                    loss = elem_loss.mean()
                elif isinstance(criterion, nn.BCEWithLogitsLoss) and getattr(criterion, 'reduction', 'mean') == 'none':
                    loss = criterion(logits[valid], target[valid].float()).mean()
                else:
                    bce = nn.BCEWithLogitsLoss(reduction='none')
                    loss = bce(logits[valid], target[valid].float()).mean()

                if return_cm:
                    acc, bal_acc, f1, f1w, rec_pos, rec_neg, prec_pos, prec_neg, cm = \
                        compute_metrics(logits[valid], target[valid], num_labels=1, return_cm=True)
                    cumulative_cm = cm if cumulative_cm is None else (cumulative_cm + cm)
                else:
                    acc, bal_acc, f1, f1w, rec_pos, rec_neg, prec_pos, prec_neg = \
                        compute_metrics(logits[valid], target[valid], num_labels=1)

                batch_size = sample.shape[0]
                metric_logger.update(loss=float(loss.item()))
                metric_logger.meters['acc'].update(acc, n=batch_size)
                metric_logger.meters['f1'].update(f1, n=batch_size)
                metric_logger.meters['f1_w'].update(f1w, n=batch_size)
                metric_logger.meters['bal_acc'].update(bal_acc, n=batch_size)
                metric_logger.meters['rec_pos'].update(rec_pos, n=batch_size)
                metric_logger.meters['rec_neg'].update(rec_neg, n=batch_size)
                metric_logger.meters['prec_pos'].update(prec_pos, n=batch_size)
                metric_logger.meters['prec_neg'].update(prec_neg, n=batch_size)

            else:
                sample = sample.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if model.num_labels >= 2:
                    target = target.squeeze().long()

                with torch.cuda.amp.autocast():
                    output, attention = model(sample, return_attn=True)
                    loss = criterion(output, target)

                if return_cm:
                    if model.num_labels == 1:
                        acc, bal_acc, f1, f1w, rec_pos, rec_neg, prec_pos, prec_neg, cm = compute_metrics(output, target, num_labels=model.num_labels, return_cm=return_cm)
                    else:
                        acc, f1, f1w, cm = compute_metrics(output, target, num_labels=model.num_labels, return_cm=return_cm)
                    
                    if cumulative_cm is None:
                        cumulative_cm = cm
                    else:
                        cumulative_cm += cm

                else:
                    if model.num_labels == 1:
                        acc, bal_acc, f1, f1w, rec_pos, rec_neg, prec_pos, prec_neg = compute_metrics(output, target, num_labels=model.num_labels)
                    else:
                        acc, f1, f1w = compute_metrics(output, target, num_labels=model.num_labels, return_cm=return_cm)

                batch_size = sample.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc'].update(acc, n=batch_size)
                metric_logger.meters['f1'].update(f1, n=batch_size)
                metric_logger.meters['f1_w'].update(f1w, n=batch_size)
                
                if model.num_labels == 1:
                    metric_logger.meters['bal_acc'].update(bal_acc, n=batch_size)
                    metric_logger.meters['rec_pos'].update(rec_pos, n=batch_size)
                    metric_logger.meters['rec_neg'].update(rec_neg, n=batch_size)
                    metric_logger.meters['prec_pos'].update(prec_pos, n=batch_size)
                    metric_logger.meters['prec_neg'].update(prec_neg, n=batch_size)
        
    metric_logger.synchronize_between_processes()
    print('* Accuracy {acc.global_avg:.3f} F1 Score (binary/macro) {f1.global_avg:.3f} loss: {losses.global_avg:.3f}'
          .format(acc=metric_logger.acc, f1=metric_logger.f1, losses=metric_logger.loss))

    if return_cm:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, cumulative_cm
    else:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_challenge(data_loader, model, device, args=None):
    """
    Runs AR generation on the loader, collects per-patient probs & predictions,
    and returns the official Challenge metrics.
    """
    model.eval()

    cohort_labels = []
    cohort_probs  = []
    cohort_preds  = []

    threshold = getattr(args, 'threshold', 0.5)

    for batch in data_loader:
        X_pad, y_pad, lengths = batch
        X_pad = X_pad.to(device, non_blocking=True)
        y_pad = y_pad.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        logits, _yhat = model.generate(X_pad, lengths, threshold=threshold, return_logits=True)  # [B,T], [B,T]
        probs = torch.sigmoid(logits)

        B, T = y_pad.shape
        for b in range(B):
            L = int(lengths[b].item())
            if L <= 0:
                continue
            cohort_labels.append(y_pad[b, :L].detach().cpu().numpy().astype(int))
            cohort_probs.append(probs[b, :L].detach().cpu().numpy().astype(float))
            cohort_preds.append(_yhat[b, :L].detach().cpu().numpy().astype(int))

    auroc, auprc, acc, f1, util = evaluate_sepsis_from_lists(cohort_labels, cohort_preds, cohort_probs)
    print(f"[challenge] AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Acc={acc:.4f}  F1={f1:.4f}  Utility={util:.4f}")

    return {
        "auroc": auroc, "auprc": auprc, "acc": acc, "f1": f1, "utility": util
    }


def compute_metrics(outputs, targets, num_labels, return_cm=False):
    if num_labels >= 2:
        probas = F.softmax(outputs, dim=1)
        preds = torch.argmax(probas, dim=1)

        accuracy = accuracy_score(targets.cpu(), preds.cpu()) * 100.
        f1 = f1_score(targets.cpu(), preds.cpu(), average='macro', zero_division=0)
        f1_weighted = f1_score(targets.cpu(), preds.cpu(), average='weighted', zero_division=0)
        if return_cm:
            cm = confusion_matrix(targets.cpu(), preds.cpu())
            return accuracy, f1, f1_weighted, cm
        else:
            return accuracy, f1, f1_weighted

    else:
        probas = torch.sigmoid(outputs).squeeze()
        preds = (probas > 0.5).long()

        accuracy = accuracy_score(targets.cpu(), preds.cpu()) * 100.
        balanced_accuracy = balanced_accuracy_score(targets.cpu(), preds.cpu()) * 100.
        f1 = f1_score(targets.cpu(), preds.cpu(), average='binary', zero_division=0)
        f1_weighted = f1_score(targets.cpu(), preds.cpu(), average='weighted', zero_division=0)
        recall_positive = recall_score(targets.cpu(), preds.cpu(), pos_label=1, average='binary', zero_division=0)
        recall_negative = recall_score(targets.cpu(), preds.cpu(), pos_label=0, average='binary', zero_division=0)
        precision_positive = precision_score(targets.cpu(), preds.cpu(), pos_label=1, average='binary', zero_division=0)
        precision_negative = precision_score(targets.cpu(), preds.cpu(), pos_label=0, average='binary', zero_division=0)

        if return_cm:
            cm = confusion_matrix(targets.cpu(), preds.cpu())
            return accuracy, balanced_accuracy, f1, f1_weighted, recall_positive, recall_negative, precision_positive, precision_negative, cm
        else:
            return accuracy, balanced_accuracy, f1, f1_weighted, recall_positive, recall_negative, precision_positive, precision_negative
