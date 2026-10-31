"""Augmentation pipelines.

Two styles, kept apart because the two pipelines are not numerically
interchangeable and merging them would change training behaviour.
"""

from __future__ import annotations

import cv2
import albumentations as A
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2

#: Probability of each random flip during training.
FLIP_PROBABILITY = 0.2

#: ImageNet statistics, used by the albumentations pipeline.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def albumentations_style(image_size: int, train: bool = True, seed: int = None) -> A.Compose:
    """CVC-ClinicDB and Kvasir-SEG.

    ``seed`` is required for repeatable flips: albumentations >= 2.0 keeps its
    own generator and ignores the global NumPy seed.
    """
    steps = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        A.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST),
    ]
    if train:
        steps.extend([
            A.HorizontalFlip(p=FLIP_PROBABILITY),
            A.VerticalFlip(p=FLIP_PROBABILITY),
        ])
    steps.append(ToTensorV2())
    return A.Compose(steps, seed=seed)


def torchvision_style(image_size: int, mean: float = 0.5, std: float = 0.5,
                      train: bool = True):
    """EDD 2020, PolypGen and SUN-SEG."""
    image_steps = [transforms.Resize((image_size, image_size), antialias=True)]
    if train:
        image_steps.extend([
            transforms.RandomHorizontalFlip(p=FLIP_PROBABILITY),
            transforms.RandomVerticalFlip(p=FLIP_PROBABILITY),
        ])
    image_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    label = transforms.Resize((image_size, image_size), antialias=True)
    return transforms.Compose(image_steps), label
