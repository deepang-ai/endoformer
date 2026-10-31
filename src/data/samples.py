"""Per-dataset enumeration of ``(image, masks)`` pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

#: Lesion classes used by EDD 2020, in channel order.
EDD_CLASSES = ("BE", "cancer", "HGD", "polyp", "suspicious")


@dataclass(frozen=True)
class Sample:
    """One image and the masks that belong to it.

    Binary datasets use the single key ``"foreground"``; multi-label datasets
    use the class names.
    """

    image: Path
    masks: Dict[str, Path]


def _list_dir(path: Path) -> list:
    """Directory contents as sorted paths, skipping hidden entries."""
    return sorted(
        p for p in Path(path).iterdir()
        if not p.name.startswith(".")
    )


def list_flat_pairs(image_dir: Path, mask_dir: Path) -> list:
    """Image/mask pairs sharing a filename (CVC-ClinicDB, Kvasir-SEG)."""
    return [
        Sample(image=p, masks={"foreground": Path(mask_dir) / p.name})
        for p in _list_dir(image_dir)
        if p.is_file()
    ]


def list_edd(image_root: Path, mask_root: Path) -> list:
    """EDD 2020: one image, up to five class masks named ``<stem>_<class>.tif``."""
    image_root, mask_root = Path(image_root), Path(mask_root)
    samples = []
    for image_path in image_root.iterdir():
        stem = image_path.stem
        masks = {}
        for class_name in EDD_CLASSES:
            mask_path = mask_root / f"{stem}_{class_name}.tif"
            if mask_path.exists():
                masks[class_name] = mask_path
        samples.append(Sample(image=image_path, masks=masks))
    return samples


def list_polypgen(root: Path) -> list:
    """PolypGen: one directory per centre, each with ``images/`` and ``masks/``."""
    root = Path(root)
    samples = []
    for centre in root.iterdir():
        image_dir, mask_dir = centre / "images", centre / "masks"
        if not image_dir.is_dir():
            continue
        for image_path in image_dir.iterdir():
            samples.append(Sample(
                image=image_path,
                masks={"foreground": mask_dir / f"{image_path.stem}_mask.jpg"},
            ))
    return samples


def list_sunseg_split(frame_dir: Path, gt_dir: Path) -> list:
    """SUN-SEG: clips as subdirectories of ``Frame/`` and ``GT/``."""
    frame_dir, gt_dir = Path(frame_dir), Path(gt_dir)
    samples = []
    for clip in frame_dir.iterdir():
        clip_gt = gt_dir / clip.name
        if not clip.is_dir():
            continue
        for image_path in clip.iterdir():
            samples.append(Sample(
                image=image_path,
                masks={"foreground": clip_gt / f"{image_path.stem}.png"},
            ))
    return samples


def list_sunseg(root: Path) -> list:
    """SUN-SEG training split, used with :func:`list_sunseg_unseen` for evaluation."""
    return list_sunseg_split(Path(root) / "TrainDataset" / "Frame",
                             Path(root) / "TrainDataset" / "GT")


def list_sunseg_unseen(root: Path) -> list:
    """SUN-SEG-Hard (Unseen) test split."""
    base = Path(root) / "TestHardDataset" / "Unseen"
    return list_sunseg_split(base / "Frame", base / "GT")
