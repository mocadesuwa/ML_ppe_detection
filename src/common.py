"""Shared paths, configuration and experiment provenance."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_yaml(path: Path) -> dict:
    import yaml
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def dataset_config(path: Path) -> dict:
    config = load_yaml(path)
    base = Path(config.get("path", "."))
    base = (path.parent / base).resolve() if not base.is_absolute() else base.resolve()
    config["path"] = str(base)
    for split in SPLITS:
        value = config.get(split)
        if not isinstance(value, str):
            raise ValueError(f"Expected an image directory for {split}")
        directory = Path(value)
        config[split] = str((base / directory).resolve() if not directory.is_absolute() else directory.resolve())
    names = config["names"]
    config["names"] = {i: str(n) for i, n in enumerate(names)} if isinstance(names, list) else {int(i): str(n) for i, n in names.items()}
    if sorted(config["names"]) != list(range(len(config["names"]))):
        raise ValueError("Class IDs must be contiguous, starting at zero")
    return config


def local_caches() -> None:
    for name, subdirectory in (("YOLO_CONFIG_DIR", "ultralytics"), ("MPLCONFIGDIR", "matplotlib"), ("TORCH_HOME", "torch")):
        path = ROOT / ".cache" / subdirectory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def run_directory(project: Path, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Use letters, digits, underscores and hyphens for an experiment name")
    path = project / name
    if path.exists():
        raise FileExistsError(f"Experiment already exists; choose a new name: {path}")
    path.mkdir(parents=True)
    return path


def select_device(requested: str) -> str | int:
    import torch
    if requested == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Prepare a CUDA PyTorch environment, or explicitly use --device cpu for a small check.")
        return 0
    if requested == "cpu":
        return requested
    index = int(requested)
    if not torch.cuda.is_available() or not 0 <= index < torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {index} is unavailable")
    return index


def dataset_fingerprint(config: dict) -> str:
    """Fingerprint images, labels, names and split membership for fair comparisons."""
    digest = hashlib.sha256(json.dumps(config["names"], sort_keys=True).encode())
    for split in SPLITS:
        directory = Path(config[split])
        for image in sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            label = Path(config["path"]) / "labels" / split / (image.stem + ".txt")
            digest.update(f"{split}/{image.name}".encode("utf-8"))
            digest.update(sha256(image).encode())
            digest.update(sha256(label).encode())
    return digest.hexdigest()


def require_review(config: dict) -> None:
    review_path = Path(config["path"]) / "quality_review.json"
    if not review_path.is_file():
        raise RuntimeError("Inspect the label sheets and record the data review first: python -m src.review_dataset --help")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("dataset_fingerprint") != dataset_fingerprint(config):
        raise RuntimeError("The dataset changed after review. Recheck labels and update the review record.")
