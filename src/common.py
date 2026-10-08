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
    names = config.get("names")
    if not isinstance(names, (list, dict)) or not names:
        raise ValueError("Class names must be a non-empty list or mapping")
    normalized = {}
    for key, name in (enumerate(names) if isinstance(names, list) else names.items()):
        if isinstance(key, bool) or not (isinstance(key, int) or isinstance(key, str) and re.fullmatch(r"[0-9]+", key)):
            raise ValueError(f"Class ID must be an integer or digit string: {key}")
        index = int(key)
        if index in normalized:
            raise ValueError(f"Repeated class ID after normalization: {key}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Class names must be non-empty strings")
        normalized[index] = name.strip()
    config["names"] = normalized
    if sorted(config["names"]) != list(range(len(config["names"]))):
        raise ValueError("Class IDs must be contiguous, starting at zero")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("Class names must be distinct")
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
    """Bind content, class names, frozen metadata and split membership."""
    digest = hashlib.sha256(b"ppe-dataset-fingerprint-v2")
    digest.update(json.dumps(config["names"], sort_keys=True).encode())
    base = Path(config["path"])
    for relative in ("manifest.json", *(f"splits/{split}.txt" for split in SPLITS)):
        path = base / relative
        digest.update(relative.encode())
        digest.update(sha256(path).encode() if path.is_file() else b"missing")
    for split in SPLITS:
        directory = Path(config[split])
        for image in sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            digest.update(f"images/{split}/{image.name}".encode("utf-8"))
            digest.update(sha256(image).encode())
        label_dir = base / "labels" / split
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")
        for label in sorted(p for p in label_dir.glob("*.txt") if p.is_file()):
            digest.update(f"labels/{split}/{label.name}".encode("utf-8"))
            digest.update(sha256(label).encode())
    return digest.hexdigest()


def require_review(config: dict) -> None:
    review_path = Path(config["path"]) / "quality_review.json"
    if not review_path.is_file():
        raise RuntimeError("Inspect the label sheets and record the data review first: python -m src.review_dataset --help")
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise RuntimeError("Cannot read the data review record. Recheck data and record the review again.") from error
    if not isinstance(review, dict) or not isinstance(review.get("dataset_fingerprint"), str):
        raise RuntimeError("Invalid data review record. Recheck data and record the review again.")
    try:
        fingerprint = dataset_fingerprint(config)
    except OSError as error:
        raise RuntimeError("Cannot verify the reviewed dataset. Check missing or unreadable files.") from error
    if review["dataset_fingerprint"] != fingerprint:
        raise RuntimeError("The dataset changed after review. Recheck labels and update the review record.")
