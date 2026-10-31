"""Configuration loading and command-line overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from easydict import EasyDict

#: Required top-level sections of the experiment config.
SECTIONS = ("run", "data", "model")


@dataclass
class Overrides:
    """Command-line overrides applied on top of the YAML file."""

    dataset: str = None
    data_root: str = None
    epochs: int = None
    lr: float = None
    optimizer: str = None
    seed: int = None
    checkpoint: str = None
    pretrained: str = None
    resume: bool = None


def load_config(path: str, overrides: Overrides = None) -> EasyDict:
    """Read the config file and apply overrides, returning a fresh object."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    missing = [section for section in SECTIONS if section not in raw]
    if missing:
        raise KeyError(
            f"{path} is missing the section(s) {missing}. "
            f"Expected {list(SECTIONS)}.")

    config = EasyDict(raw)
    if overrides is not None:
        _apply(config, overrides)
    return config


def _apply(config, overrides: Overrides) -> None:
    """Copy every non-None override onto ``config``."""
    simple = {
        "epochs": ("run", "epochs"),
        "lr": ("run", "lr"),
        "optimizer": ("run", "optimizer"),
        "seed": ("run", "seed"),
        "checkpoint": ("run", "checkpoint"),
        "dataset": ("run", "dataset"),
        "data_root": ("data", "root"),
        "pretrained": ("model", "pretrained"),
    }
    for field, (section, key) in simple.items():
        value = getattr(overrides, field)
        if value is not None:
            config[section][key] = value
    if overrides.resume is not None:
        config.run.resume = overrides.resume


def checkpoint_dir(config, split: str) -> Path:
    """Path of the ``best`` or ``checkpoint`` directory for this run."""
    if split not in ("best", "checkpoint"):
        raise ValueError(f"Unknown split {split!r}; expected 'best' or 'checkpoint'")
    return Path.cwd() / "model_store" / config.run.checkpoint / split


def checkpoint_file(config, split: str = "best") -> Path:
    """Weights file written by ``accelerator.save_state``.

    The ``pytorch_model.bin`` beside it may hold uninitialised tensors; read
    ``model.safetensors`` instead.
    """
    return checkpoint_dir(config, split) / "model.safetensors"
