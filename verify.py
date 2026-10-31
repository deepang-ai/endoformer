"""Evaluate a trained EndoFormer checkpoint on the validation split."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import monai
from accelerate import Accelerator
from monai.utils import ensure_tuple_rep
from objprint import objstr

from src import utils
from src.config import Overrides, checkpoint_file, load_config
from src.data import build_dataloaders
from src.engine import class_names_for, evaluate
from src.metrics import create_segmentation_metrics
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to the experiment config')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Weights file to evaluate (defaults to the best checkpoint)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override run.dataset')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override data.root')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, Overrides(dataset=args.dataset,
                                                data_root=args.data_root))
    utils.same_seeds(config.run.seed)

    logging_dir = f"{os.getcwd()}/logs/{config.run.checkpoint}{datetime.now()}"
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"],
                              project_dir=logging_dir)
    utils.Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(Path(__file__).stem)
    accelerator.print(objstr(config))

    accelerator.print('Load data...')
    _, val_loader, spec = build_dataloaders(config, config.run.seed)

    accelerator.print('Build model...')
    model = build_model(config, spec.num_classes)

    weights = Path(args.checkpoint) if args.checkpoint else checkpoint_file(config)
    if not weights.exists():
        raise FileNotFoundError(
            f"No checkpoint at {weights}.\n"
            f"Train a model first (python train.py) or pass --checkpoint <path>.")
    model = utils.load_pretrain_model(weights, model, accelerator)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(config.data.image_size, 2), overlap=0.5,
        sw_device=accelerator.device, device=accelerator.device)
    metrics = create_segmentation_metrics(spec.score_background)
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True),
        monai.transforms.AsDiscrete(threshold=0.5),
    ])
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5,
                                          to_onehot_y=False, sigmoid=True),
    }

    model, val_loader = accelerator.prepare(model, val_loader)

    mean_dice, results, _ = evaluate(
        model, loss_functions, val_loader, inference, metrics, post_trans,
        accelerator, epoch=0, step=0, num_epochs=1,
        score_background=spec.score_background,
        class_names=class_names_for(spec))

    accelerator.print(f"Dice: {mean_dice.item():.4f}")
    for key, value in results.items():
        accelerator.print(f"  {key}: {value:.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
