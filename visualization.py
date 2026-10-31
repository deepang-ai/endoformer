"""Render predicted masks and error maps for a set of images.

Loads the best checkpoint saved by ``train.py`` and runs inference on every
image under ``visualization.img_path``. Binary masks are written next to the
ground truth, contour and error overlays to ``visualization.output_path``.
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import albumentations as A
import cv2
import imageio.v2 as imageio
import monai
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from albumentations.pytorch import ToTensorV2
from monai.utils import ensure_tuple_rep
from objprint import objstr
from PIL import Image

from src import utils
from src.config import Overrides, checkpoint_file, load_config
from src.data import get_spec
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.model import build_model

#: Colour of the correct and incorrect regions in the error overlay.
CORRECT_COLOR = (0, 255, 0)
INCORRECT_COLOR = (255, 0, 0)

#: Weight of the original image in the error overlay.
OVERLAY_ALPHA = 0.7


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to the experiment config')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Weights file to use (defaults to the best checkpoint)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override run.dataset')
    return parser.parse_args()


def list_images(folder: str) -> list:
    return [Path(root) / name
            for root, _, files in os.walk(folder) for name in files]


def prepare_input(image_path: Path, image_size: int):
    """Load an image at network resolution, keeping the original size for output."""
    pipeline = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        A.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST),
        ToTensorV2(),
    ])
    original = A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    size = original(image=image, mask=image)['image'].shape[-2:]
    return pipeline(image=image, mask=image)['image'].unsqueeze(0), tuple(size)


@torch.no_grad()
def write_masks(model, inferer, post_trans, image_paths, output_dir: Path,
                image_size: int, accelerator) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for image_path in image_paths:
        image, original_size = prepare_input(image_path, image_size)
        logits = post_trans(inferer(image.to(accelerator.device), model))
        logits = F.interpolate(logits, size=original_size, mode='nearest')
        mask = (logits.cpu().squeeze().numpy() * 255).astype(np.uint8)
        Image.fromarray(mask).convert('L').save(
            output_dir / f'{image_path.stem}{image_path.suffix}')
        accelerator.print(f'{output_dir / image_path.stem} saved')


def write_overlays(image_paths, ground_truth_dir: Path, mask_dir: Path,
                   output_dir: Path) -> None:
    """Draw predicted contours, plus an error map when ground truth exists."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for image_path in image_paths:
        prediction_path = mask_dir / f'{image_path.stem}{image_path.suffix}'
        if not prediction_path.exists():
            print(f'No prediction for {image_path.name}, skipped')
            continue

        image = imageio.imread(image_path)
        predicted = cv2.cvtColor(cv2.cvtColor(imageio.imread(prediction_path),
                                             cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

        ground_truth_path = ground_truth_dir / f'{image_path.stem}{image_path.suffix}'
        if ground_truth_path.exists():
            ground_truth = cv2.cvtColor(cv2.cvtColor(imageio.imread(ground_truth_path),
                                                     cv2.COLOR_RGB2BGR),
                                        cv2.COLOR_BGR2GRAY)
            error_map = np.zeros_like(image, dtype=np.uint8)
            predicted_foreground = predicted == 255
            matches = (predicted == ground_truth) & predicted_foreground
            error_map[matches, 1] = CORRECT_COLOR[1]
            error_map[~matches & predicted_foreground, 0] = INCORRECT_COLOR[0]
            overlay = cv2.addWeighted(image, OVERLAY_ALPHA, error_map,
                                      1 - OVERLAY_ALPHA, 0)
            imageio.imwrite(output_dir / f'{image_path.stem}_error.png', overlay)

        _, binary = cv2.threshold(predicted, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 223, 146), 2)
        imageio.imwrite(output_dir / f'{image_path.stem}.png', image)
        print(f'{image_path.name} overlay written')


def main() -> int:
    args = parse_args()
    config = load_config(args.config, Overrides(dataset=args.dataset))
    utils.same_seeds(config.run.seed)

    logging_dir = f"{os.getcwd()}/logs/{config.run.checkpoint}{datetime.now()}"
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"],
                              project_dir=logging_dir)
    utils.Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(Path(__file__).stem)
    accelerator.print(objstr(config))

    spec = get_spec(config.run.dataset)

    accelerator.print('Build model...')
    model = build_model(config, spec.num_classes)
    weights = Path(args.checkpoint) if args.checkpoint else checkpoint_file(config)
    if not weights.exists():
        raise FileNotFoundError(
            f"No checkpoint at {weights}.\n"
            f"Train a model first (python train.py) or pass --checkpoint <path>.")
    model = utils.load_pretrain_model(weights, model, accelerator)
    model = accelerator.prepare(model)
    model.eval()

    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(config.data.image_size, 2), overlap=0.5,
        sw_device=accelerator.device, device=accelerator.device)
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True),
        monai.transforms.AsDiscrete(threshold=0.5),
    ])

    vis = config.visualization
    image_paths = list_images(vis.img_path)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found under {vis.img_path}. "
            f"Put the images to segment there.")

    accelerator.print(f'Found {len(image_paths)} image(s)')
    mask_dir = Path(vis.mask_path) / spec.name
    write_masks(model, inferer, post_trans, image_paths, mask_dir,
                config.data.image_size, accelerator)
    write_overlays(image_paths, Path(vis.mask_path) / 'GT',
                   mask_dir, Path(vis.output_path) / spec.name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
