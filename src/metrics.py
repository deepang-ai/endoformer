"""Segmentation metrics."""

from __future__ import annotations

from typing import Dict

import monai


def create_segmentation_metrics(score_background: bool) -> Dict[str, monai.metrics.CumulativeIterationMetric]:
    """Build the metric suite reported in the paper.

    For single-channel outputs MONAI treats both values of ``score_background``
    identically, so the flag only matters for multi-label datasets.
    """
    reduction = monai.utils.MetricReduction.MEAN_BATCH
    return {
        "dice": monai.metrics.DiceMetric(
            include_background=score_background,
            reduction=reduction, get_not_nans=True),
        "mIoU": monai.metrics.MeanIoU(include_background=score_background),
        "F1": monai.metrics.ConfusionMatrixMetric(
            include_background=score_background, metric_name="f1 score"),
        "Precision": monai.metrics.ConfusionMatrixMetric(
            include_background=score_background, metric_name="precision"),
        "Recall": monai.metrics.ConfusionMatrixMetric(
            include_background=score_background, metric_name="recall"),
        "HD95": monai.metrics.HausdorffDistanceMetric(
            percentile=95, include_background=score_background,
            reduction=reduction, get_not_nans=True),
    }


def reset_metrics(metrics: Dict[str, monai.metrics.CumulativeIterationMetric]) -> None:
    for metric in metrics.values():
        metric.reset()


def aggregate_metrics(metrics, accelerator, score_background: bool,
                      class_names=None) -> Dict[str, float]:
    """Reduce accumulated metrics to a flat ``{name: value}`` dict.

    ``class_names`` extends each entry with per-class values; pass ``None`` for
    single-channel data, where only the mean is meaningful.
    """
    results = {}
    for name, metric in metrics.items():
        values = metric.aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            values = accelerator.reduce(values) / accelerator.num_processes
        metric.reset()
        results[f"mean {name}"] = float(values.mean())
        if class_names and values.numel() > 1:
            for index, class_name in enumerate(class_names):
                results[f"{class_name} {name}"] = float(values[index])
    return results
