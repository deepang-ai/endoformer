"""Dataset declarations and loader construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import (
    AlbumentationsBinaryDataset,
    BinarySegDataset,
    MultiLabelSegDataset,
)
from src.data.samples import (
    list_edd,
    list_flat_pairs,
    list_polypgen,
    list_sunseg,
    list_sunseg_unseen,
)
from src.data.transforms import albumentations_style, torchvision_style


def _cvc(root: Path) -> list:
    return list_flat_pairs(Path(root) / "Original", Path(root) / "GroundTruth")


def _kvasir(root: Path) -> list:
    return list_flat_pairs(Path(root) / "images", Path(root) / "masks")


def _edd(root: Path) -> list:
    return list_edd(Path(root) / "originalImages", Path(root) / "masks")


@dataclass(frozen=True)
class DatasetSpec:
    """What the loaders and the metrics need to know about a dataset.

    ``score_background`` is ``False`` everywhere: MONAI ignores it for
    single-channel outputs, and all five EDD classes are lesions, so no dataset
    here has a background class to score. ``has_official_split`` marks the
    datasets that bring their own partitions (SUN-SEG); the others are split by
    ``data.train_ratio``.
    """

    name: str
    num_classes: int
    score_background: bool
    build_samples: Callable[[Path], list]
    has_official_split: bool
    uses_albumentations: bool


DATASETS = {
    "CVC_ClinicDB": DatasetSpec(
        name="CVC_ClinicDB", num_classes=1, score_background=False,
        build_samples=_cvc, has_official_split=False, uses_albumentations=True),
    "Kvasir_SEG": DatasetSpec(
        name="Kvasir_SEG", num_classes=1, score_background=False,
        build_samples=_kvasir, has_official_split=False, uses_albumentations=True),
    "EDD_seg": DatasetSpec(
        name="EDD_seg", num_classes=5, score_background=False,
        build_samples=_edd, has_official_split=False, uses_albumentations=False),
    "PolypGen": DatasetSpec(
        name="PolypGen", num_classes=1, score_background=False,
        build_samples=list_polypgen, has_official_split=False,
        uses_albumentations=False),
    "Sun_seg": DatasetSpec(
        name="Sun_seg", num_classes=1, score_background=False,
        build_samples=list_sunseg, has_official_split=True,
        uses_albumentations=False),
}


def get_spec(name: str) -> DatasetSpec:
    """Look up a dataset by its config name."""
    try:
        return DATASETS[name]
    except KeyError:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"Unknown dataset {name!r}. Set run.dataset to one of: {supported}") from None


def _split(samples: list, train_ratio: float) -> tuple:
    """Split by position, keeping the enumeration order stable."""
    boundary = int(len(samples) * train_ratio)
    return samples[:boundary], samples[boundary:]


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


def _make_loader(dataset, batch_size: int, num_workers: int,
                 shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def build_dataloaders(config, seed: int = 50) -> tuple:
    """Return ``(train_loader, val_loader, spec)`` for ``config.run.dataset``."""
    spec = get_spec(config.run.dataset)
    data = config.data
    root = Path(data.root)

    if spec.has_official_split:
        train_samples = spec.build_samples(root)
        val_samples = list_sunseg_unseen(root)
    else:
        train_samples, val_samples = _split(spec.build_samples(root),
                                            data.train_ratio)

    if not train_samples:
        raise FileNotFoundError(
            f"No training samples found for {spec.name} under {root}. "
            f"Check data.root and the layout documented in README.md.")
    if not val_samples:
        raise FileNotFoundError(
            f"No validation samples found for {spec.name} under {root}.")

    if spec.uses_albumentations:
        # ``seed`` is required, not optional; see ``albumentations_style``.
        train_dataset = AlbumentationsBinaryDataset(
            train_samples,
            albumentations_style(data.image_size, train=True, seed=seed),
            train=True)
        val_dataset = AlbumentationsBinaryDataset(
            val_samples,
            albumentations_style(data.image_size, train=False, seed=seed),
            train=False)
    else:
        dataset_cls = MultiLabelSegDataset if spec.num_classes > 1 else BinarySegDataset
        train_image, train_label = torchvision_style(
            data.image_size, data.mean, data.std, train=True)
        val_image, val_label = torchvision_style(
            data.image_size, data.mean, data.std, train=False)
        train_dataset = dataset_cls(train_samples, train_image, train_label)
        val_dataset = dataset_cls(val_samples, val_image, val_label)

    return (
        _make_loader(train_dataset, data.batch_size, data.num_workers, True, seed),
        _make_loader(val_dataset, data.batch_size, data.num_workers, False, seed),
        spec,
    )
