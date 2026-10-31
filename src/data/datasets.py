"""Dataset classes over the samples enumerated in :mod:`src.data.samples`."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import pil_to_tensor

from src.data.samples import EDD_CLASSES, Sample

#: Probability of applying LAB colour transfer to the training image.
COLOR_TRANSFER_PROBABILITY = 0.7

#: Guards the division when transferring colour statistics.
EPSILON = 1e-6


class BinarySegDataset(Dataset):
    """Image plus a single foreground mask, transformed with torchvision."""

    def __init__(self, samples: Sequence[Sample],
                 image_transform: Callable, label_transform: Callable,
                 loader: Callable = default_loader) -> None:
        self.samples = list(samples)
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.loader = loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self.loader(sample.image)
        mask = self.loader(sample.masks["foreground"]).convert("L")
        label = (pil_to_tensor(mask) == 255.0).float()
        return self.image_transform(image), self.label_transform(label)


class MultiLabelSegDataset(Dataset):
    """Image plus one binary mask per lesion class (EDD 2020).

    A class without an annotation file contributes an all-zero channel.
    """

    def __init__(self, samples: Sequence[Sample],
                 image_transform: Callable, label_transform: Callable,
                 classes: Sequence[str] = EDD_CLASSES,
                 loader: Callable = default_loader) -> None:
        self.samples = list(samples)
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.classes = tuple(classes)
        self.loader = loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self.loader(sample.image)

        label = torch.zeros(len(self.classes), image.height, image.width)
        for class_name, mask_path in sample.masks.items():
            mask = self.loader(mask_path).convert("L")
            label[self.classes.index(class_name)] = (
                pil_to_tensor(mask) == 255.0).float()

        return self.image_transform(image), self.label_transform(label)


class AlbumentationsBinaryDataset(Dataset):
    """Image plus a single mask, with LAB colour transfer during training.

    Used for CVC-ClinicDB and Kvasir-SEG.
    """

    def __init__(self, samples: Sequence[Sample], transform,
                 train: bool = True) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.train = train

        stem_is_numeric = [s.image.stem.isdigit() for s in self.samples]
        self.numeric_stem = [s.image for s, flag in zip(self.samples, stem_is_numeric) if flag]
        self.other_stem = [s.image for s, flag in zip(self.samples, stem_is_numeric) if not flag]

    def __len__(self) -> int:
        return len(self.samples)

    def _donor_image(self, index: int) -> Path:
        """Pick the image whose colour statistics replace the current ones.

        The draw is skipped when both filename styles are not present, so the
        RNG advances differently. Keep the order of these checks.
        """
        if self.other_stem and np.random.rand() < COLOR_TRANSFER_PROBABILITY:
            pool = self.other_stem
        elif self.numeric_stem:
            pool = self.numeric_stem
        else:
            return self.samples[index].image
        return pool[index % len(pool)]

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = cv2.imread(str(sample.image))

        if self.train:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            donor = cv2.cvtColor(cv2.imread(str(self._donor_image(index))),
                                 cv2.COLOR_BGR2LAB)
            mean = image.mean(axis=(0, 1), keepdims=True)
            std = image.std(axis=(0, 1), keepdims=True)
            donor_mean = donor.mean(axis=(0, 1), keepdims=True)
            donor_std = donor.std(axis=(0, 1), keepdims=True)
            image = np.uint8(
                (image - mean) / np.maximum(std, EPSILON) * donor_std + donor_mean)
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(sample.masks["foreground"]), cv2.IMREAD_GRAYSCALE) / 255.0
        pair = self.transform(image=image, mask=mask)
        return pair["image"], pair["mask"].unsqueeze(0)
