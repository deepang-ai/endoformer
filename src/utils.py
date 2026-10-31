"""Shared helpers: seeding, checkpoint I/O and logging."""

from __future__ import annotations

import os
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn


@dataclass
class TrainState:
    """Where an interrupted run left off."""

    start_epoch: int = 0
    step: int = 0
    best_acc: torch.Tensor = field(default_factory=lambda: torch.tensor(0))
    best_class: List = field(default_factory=list)


def load_state_dict(path) -> OrderedDict:
    """Read a state dict from a local path or a remote URL."""
    source = str(path)
    if source.startswith('http'):
        return torch.hub.load_state_dict_from_url(source, map_location='cpu')
    return torch.load(source, map_location='cpu')


def load_pretrain_model(path, model: nn.Module, accelerator: Accelerator) -> nn.Module:
    """Load weights into ``model``, preferring ``model.safetensors`` over a
    ``pytorch_model.bin`` path.

    The ``.bin`` written by ``accelerator.save_state`` may hold uninitialised
    tensors.
    """
    try:
        safetensors_path = str(path).replace('pytorch_model.bin', 'model.safetensors')
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        else:
            state_dict = load_state_dict(path)
        model.load_state_dict(state_dict)
        accelerator.print('Model loaded.')
    except Exception as error:
        accelerator.print(error)
        accelerator.print('Failed to load the model; continuing with current weights.')
    return model


def resume_train_state(model, checkpoint_name: str, optimizer, scheduler,
                       train_loader, accelerator: Accelerator,
                       store: str = 'model_store') -> TrainState:
    """Restore model, optimizer and scheduler from an interrupted run."""
    base = f"{os.getcwd()}/{store}/{checkpoint_name}/checkpoint"
    try:
        epoch_file = torch.load(f"{base}/epoch.pth.tar", map_location='cpu')
        state = TrainState(
            start_epoch=epoch_file['epoch'] + 1,
            step=(epoch_file['epoch'] + 1) * len(train_loader),
            best_acc=epoch_file['best_acc'],
            best_class=epoch_file.get('best_class', []))
        load_pretrain_model(f"{base}/pytorch_model.bin", model, accelerator)
        optimizer.load_state_dict(torch.load(f"{base}/optimizer.bin"))
        scheduler.load_state_dict(torch.load(f"{base}/scheduler.bin"))
        accelerator.print(
            f"Resumed from epoch {state.start_epoch}, best Dice {state.best_acc}")
        return state
    except Exception as error:
        accelerator.print(error)
        accelerator.print('No checkpoint to resume from; starting fresh.')
        return TrainState()


def same_seeds(seed: int) -> None:
    """Seed every RNG and enable deterministic kernels."""
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.multiprocessing.set_sharing_strategy('file_system')
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class Logger(object):
    """Mirror stdout and stderr to a log file."""

    def __init__(self, logdir: str):
        self.console = sys.stdout
        if logdir is not None:
            os.makedirs(logdir)
            self.log_file = open(f'{logdir}/log.txt', 'w')
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        self.console.close()
        if self.log_file is not None:
            self.log_file.close()
