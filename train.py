"""Train EndoFormer on an endoscopic polyp segmentation dataset."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import monai
import torch
import yaml
from accelerate import Accelerator
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.config import Overrides, checkpoint_dir, load_config
from src.data import build_dataloaders
from src.engine import class_names_for, evaluate, train_epoch
from src.metrics import create_segmentation_metrics
from src.model import build_model
from src.optimizer import LinearWarmupCosineAnnealingLR


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to the experiment config')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device ids to expose, e.g. "0" or "0,1"')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for initialisation, data order, workers and augmentation')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override run.dataset')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override data.root')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override run.epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override run.lr')
    parser.add_argument('--optimizer', type=str, default=None,
                        choices=['adamw', 'adam', 'sgd'],
                        help='Override run.optimizer')
    parser.add_argument('--checkpoint-name', type=str, default=None,
                        help='Override run.checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Initialise the model from a saved weights file')
    parser.add_argument('--backbone-pretrained', type=str, default=None,
                        help='Override model.pretrained (PVTv2-B2 weights)')
    parser.add_argument('--no-resume', action='store_true', default=False,
                        help='Start fresh even when run.resume is enabled')
    return parser.parse_known_args()[0]


def main() -> int:
    args = parse_args()

    overrides = Overrides(
        dataset=args.dataset, data_root=args.data_root, epochs=args.epochs,
        lr=args.lr, optimizer=args.optimizer, seed=args.seed,
        checkpoint=args.checkpoint_name, pretrained=args.backbone_pretrained,
        resume=False if args.no_resume else None)
    config = load_config(args.config, overrides)
    if args.seed is None:
        config.run.seed = 50
    if args.checkpoint_name is None:
        config.run.checkpoint = str(config.run.checkpoint).replace(
            '{seed}', str(config.run.seed))

    utils.same_seeds(config.run.seed)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging_dir = f"{os.getcwd()}/logs/{config.run.checkpoint}{datetime.now()}"
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"],
                              project_dir=logging_dir)
    utils.Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(Path(__file__).stem)
    _archive_config(args.config, config, logging_dir, accelerator)
    accelerator.print(objstr(config))

    accelerator.print('Load data...')
    train_loader, val_loader, spec = build_dataloaders(config, config.run.seed)

    accelerator.print('Build model...')
    model = build_model(config, spec.num_classes)
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(
        f"Backbone: PVTv2-{model.backbone_variant} "
        f"depths={list(model.backbone.depths)} "
        f"parameters={sum(p.numel() for p in model.backbone.parameters()) / 1e6:.2f}M")
    accelerator.print(f"Trainable/total parameters: {trainable_parameters}/{total_parameters}")
    accelerator.print(f"Output classes: {spec.num_classes} ({spec.name})")

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(config.data.image_size, 2), overlap=0.5,
        sw_device=accelerator.device, device=accelerator.device)
    train_metrics = create_segmentation_metrics(spec.score_background)
    val_metrics = create_segmentation_metrics(spec.score_background)
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True),
        monai.transforms.AsDiscrete(threshold=0.5),
    ])

    optimizer = optim_factory.create_optimizer_v2(
        model, opt=config.run.optimizer,
        weight_decay=config.run.weight_decay,
        lr=config.run.lr, betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=config.run.warmup,
        max_epochs=config.run.epochs)
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5,
                                          to_onehot_y=False, sigmoid=True),
    }

    if args.pretrained:
        model = utils.load_pretrain_model(args.pretrained, model, accelerator)

    starting_epoch, step, best_acc, best_class, best_epoch = 0, 0, torch.tensor(0), [], -1
    if config.run.resume:
        state = utils.resume_train_state(model, config.run.checkpoint,
                                         optimizer, scheduler, train_loader, accelerator)
        starting_epoch, step = state.start_epoch, state.step
        best_acc, best_class = state.best_acc, state.best_class

    joint = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)
    model, optimizer, scheduler, train_loader, val_loader = joint
    best_acc = best_acc.to(accelerator.device)

    accelerator.print('Start training!')
    for epoch in range(starting_epoch, config.run.epochs):
        step = train_epoch(model, loss_functions, train_loader, optimizer, scheduler,
                           train_metrics, post_trans, accelerator, epoch, step,
                           config.run.epochs)
        mean_dice, results, step = evaluate(
            model, loss_functions, val_loader, inference, val_metrics, post_trans,
            accelerator, epoch, step, config.run.epochs,
            score_background=spec.score_background,
            class_names=class_names_for(spec))

        if mean_dice > best_acc:
            best_acc, best_class, best_epoch = mean_dice, results, epoch
            accelerator.save_state(output_dir=str(checkpoint_dir(config, 'best')))

        accelerator.save_state(output_dir=str(checkpoint_dir(config, 'checkpoint')))
        torch.save({'epoch': epoch, 'best_acc': best_acc, 'best_class': results},
                   checkpoint_dir(config, 'checkpoint') / 'epoch.pth.tar')
        accelerator.print(
            f"Epoch [{epoch + 1}/{config.run.epochs}] best Dice: {best_acc} "
            f"current Dice: {mean_dice}")

    accelerator.print(f"Best Dice: {best_acc}")
    accelerator.print(f"Best epoch: {best_epoch}")
    return 0


def _archive_config(config_path, config, logging_dir, accelerator) -> None:
    """Keep the config that produced this run next to its log."""
    if not accelerator.is_local_main_process:
        return
    source = Path(config_path).expanduser()
    if not source.is_absolute():
        source = Path.cwd() / source
    if not source.exists():
        raise FileNotFoundError(f"Config file not found: {source}")
    target = Path(logging_dir)
    target.mkdir(parents=True, exist_ok=True)
    with open(target / 'config.yml', 'w', encoding='utf-8') as handle:
        yaml.safe_dump(_plain(config), handle, sort_keys=False,
                       allow_unicode=True)


def _plain(node):
    """Recursively convert EasyDict to plain containers for YAML dumping."""
    if isinstance(node, dict):
        return {key: _plain(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_plain(item) for item in node]
    return node


if __name__ == '__main__':
    raise SystemExit(main())
