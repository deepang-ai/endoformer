"""Model construction."""

from __future__ import annotations

from src.model.endoformer import EndoFormer

__all__ = ["EndoFormer", "build_model"]

#: Config keys whose name differs from the constructor argument.
_RENAMED = {"pretrained": "model_dir"}


def build_model(config, num_classes: int) -> EndoFormer:
    """Build EndoFormer for ``num_classes`` output channels."""
    kwargs = {_RENAMED.get(key, key): value
              for key, value in config.model.items()}
    kwargs.pop("out_channels", None)
    return EndoFormer(out_channels=num_classes, **kwargs)
