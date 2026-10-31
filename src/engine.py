"""Training and evaluation loops."""

from __future__ import annotations

from typing import Dict

import monai
import torch
from accelerate import Accelerator

from src.data.samples import EDD_CLASSES
from src.metrics import aggregate_metrics, reset_metrics


def train_epoch(model: torch.nn.Module,
                loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
                post_trans: monai.transforms.Compose,
                accelerator: Accelerator,
                epoch: int,
                step: int,
                num_epochs: int,
                loss_weights: Dict[str, float] = None) -> int:
    """Run one training epoch and return the updated global step."""
    reset_metrics(metrics)
    model.train()

    for batch_index, (images, labels) in enumerate(train_loader):
        logits = model(images)
        total_loss = 0.0
        details = ""

        for name, loss_function in loss_functions.items():
            weight = loss_weights.get(name, 1.0) if loss_weights else 1.0
            loss = loss_function(logits, labels)
            accelerator.log({f"Train/{name}": float(loss)}, step=step)
            total_loss = total_loss + weight * loss
            details += f" {name} {float(loss):1.5f} "

        predictions = post_trans(logits)
        for metric in metrics.values():
            metric(y_pred=predictions, y=labels)

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()

        accelerator.log({"Train/Total Loss": float(total_loss)}, step=step)
        accelerator.print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Training [{batch_index + 1}/{len(train_loader)}] "
            f"Loss: {float(total_loss):1.5f}{details}",
            flush=True)
        step += 1

    scheduler.step(epoch)

    results = aggregate_metrics(metrics, accelerator, score_background=False)
    summary = {f"Train/{key}": value for key, value in results.items()}
    accelerator.print(f"Epoch [{epoch + 1}/{num_epochs}] Training metric {summary}")
    accelerator.log(summary, step=epoch)
    return step


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loss_functions: Dict[str, torch.nn.modules.loss._Loss],
             val_loader: torch.utils.data.DataLoader,
             inference: monai.inferers.Inferer,
             metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
             post_trans: monai.transforms.Compose,
             accelerator: Accelerator,
             epoch: int,
             step: int,
             num_epochs: int,
             score_background: bool = False,
             class_names=None) -> tuple:
    """Run validation and return ``(mean_dice, results, step)``.

    ``mean_dice`` is the scalar used for checkpoint selection.
    """
    reset_metrics(metrics)
    model.eval()

    for batch_index, (images, labels) in enumerate(val_loader):
        logits = inference(images, model)
        total_loss = 0.0
        details = ""

        for name, loss_function in loss_functions.items():
            loss = loss_function(logits, labels)
            accelerator.log({f"Val/{name}": float(loss)}, step=step)
            total_loss = total_loss + loss
            details += f" {name} {float(loss):1.5f} "

        predictions = post_trans(logits)
        for metric in metrics.values():
            metric(y_pred=predictions, y=labels)

        accelerator.log({"Val/Total Loss": float(total_loss)}, step=step)
        accelerator.print(
            f"Validation [{batch_index + 1}/{len(val_loader)}] "
            f"Loss: {float(total_loss):1.5f}{details}",
            flush=True)
        step += 1

    results = aggregate_metrics(metrics, accelerator, score_background, class_names)
    summary = {f"Val/{key}": value for key, value in results.items()}
    accelerator.print(f"Epoch [{epoch + 1}/{num_epochs}] Validation metric {summary}")
    accelerator.log(summary, step=epoch)

    mean_dice = torch.tensor([summary["Val/mean dice"]]).to(accelerator.device)
    return mean_dice, summary, step


def class_names_for(spec) -> tuple:
    """Per-class labels for the metric breakdown, or ``None`` for binary data."""
    return EDD_CLASSES if spec.num_classes > 1 else None
