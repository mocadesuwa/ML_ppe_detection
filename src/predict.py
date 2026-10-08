"""Save boxes, names, confidence scores and images for local inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

from .common import IMAGE_SUFFIXES, config_path, local_caches, normalize_device, project_path, run_directory, select_device, sha256, utc_now, validate_run_name, write_json

PREDICT_IMAGE_SUFFIXES = IMAGE_SUFFIXES | {".avif", ".dng", ".heic", ".heif", ".jp2", ".mpo", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv", ".webm"}
MEDIA_SUFFIXES = PREDICT_IMAGE_SUFFIXES | VIDEO_SUFFIXES


def source_snapshot(source: Path) -> tuple[list[dict], str]:
    """Bind direct media files without buffering image pixels or video frames."""
    if source.is_dir():
        paths = sorted(path for path in source.iterdir() if not path.name.startswith(".") and path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES)
    elif source.is_file():
        if source.suffix.lower() not in MEDIA_SUFFIXES:
            raise ValueError(f"Unsupported local source media format: {source}")
        paths = [source]
    else:
        raise FileNotFoundError(f"Local source file or directory does not exist: {source}")
    if not paths:
        raise ValueError(f"No supported source media files directly inside: {source}")
    files = []
    for path in paths:
        if path.stat().st_size == 0:
            raise ValueError(f"Source media file is empty: {path}")
        files.append({"path": str(path.resolve()), "sha256": sha256(path)})
    digest = hashlib.sha256(json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return files, digest


def build_plan(args) -> dict:
    if isinstance(args.imgsz, bool) or not isinstance(args.imgsz, int) or args.imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    if isinstance(args.conf, bool) or not isinstance(args.conf, (int, float)) or not 0 <= args.conf <= 1 or not math.isfinite(args.conf):
        raise ValueError("conf must be a finite number between zero and one")
    validate_run_name(args.name)
    device = normalize_device(args.device)
    source = config_path(args.source, "source", project_path("."))
    weights = config_path(args.weights, "weights", project_path("."))
    project = project_path("results/predictions")
    for parent in (project, *project.parents):
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"Prediction output parent must be a directory: {parent}")
    directory = project / args.name
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Prediction already exists; choose a new name: {directory}")
    if not weights.is_file():
        raise FileNotFoundError(f"Trained weights must exist: {weights}")
    files, fingerprint = source_snapshot(source)
    return {"source": str(source), "source_files": files, "source_fingerprint": fingerprint, "weights": str(weights), "weights_sha256": sha256(weights), "project": str(project), "name": args.name, "device": device, "confidence": args.conf, "imgsz": args.imgsz}


def class_names(names: object) -> dict[int, str]:
    message = "Model class names must have distinct non-empty names and contiguous integer IDs"
    if not isinstance(names, (dict, list)) or not names:
        raise ValueError(message)
    normalized = {}
    for key, name in (enumerate(names) if isinstance(names, list) else names.items()):
        if isinstance(key, bool) or not (isinstance(key, int) or isinstance(key, str) and re.fullmatch(r"[0-9]+", key)):
            raise ValueError(message)
        index = int(key)
        if index in normalized or not isinstance(name, str) or not name.strip():
            raise ValueError(message)
        normalized[index] = name
    if sorted(normalized) != list(range(len(normalized))) or len(set(normalized.values())) != len(normalized):
        raise ValueError(message)
    return normalized


def finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Prediction {field} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"Prediction {field} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"Prediction {field} must be a finite number")
    return number


def frame_record(result, names: dict, allowed: set[Path], counts: dict[Path, int], index: int) -> tuple[Path, dict]:
    source = config_path(str(result.path) if isinstance(result.path, Path) else result.path, "result source", project_path("."))
    if source not in allowed:
        raise ValueError(f"Unexpected prediction source: {source}")
    source_index = counts.get(source, 0)
    if source.suffix.lower() in PREDICT_IMAGE_SUFFIXES and source_index:
        raise ValueError(f"Duplicate prediction result for image: {source}")
    shape = getattr(result, "orig_shape", None)
    if not isinstance(shape, (tuple, list)) or len(shape) != 2 or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape):
        raise ValueError("Prediction image shape must contain positive integer height and width")
    height, width = shape
    if class_names(result.names) != names:
        raise ValueError("Prediction class names changed from the loaded model")
    detections = []
    if result.boxes is not None:
        arrays = (result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())
        if not all(isinstance(array, list) for array in arrays) or len({len(array) for array in arrays}) != 1:
            raise ValueError("Prediction detection arrays have invalid or different lengths")
        for coordinates, category, confidence in zip(*arrays):
            if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
                raise ValueError("Prediction box must contain four coordinates")
            x1, y1, x2, y2 = [finite_number(value, "box coordinate") for value in coordinates]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("Prediction box is empty or outside the image")
            category = finite_number(category, "class ID")
            confidence = finite_number(confidence, "confidence")
            if not category.is_integer() or int(category) not in names:
                raise ValueError("Prediction class ID is not a known integer")
            if not 0 <= confidence <= 1:
                raise ValueError("Prediction confidence must be between zero and one")
            detections.append({"class_id": int(category), "name": names[int(category)], "confidence": confidence, "xyxy_pixels": [x1, y1, x2, y2]})
    return source, {"source": str(source), "frame_index": index, "source_frame_index": source_index, "image_size": {"width": width, "height": height}, "detections": detections}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict on a local image, image directory or video")
    parser.add_argument("--source", required=True, help="Local media file or directory; directory scanning does not recurse")
    parser.add_argument("--weights", default="results/baseline_01/weights/best.pt")
    parser.add_argument("--name", default="baseline_examples")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a non-negative CUDA device index")
    args = parser.parse_args()
    try:
        plan = build_plan(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    source, weights = Path(plan["source"]), Path(plan["weights"])
    try:
        device = select_device(plan["device"])
        local_caches()
        from ultralytics import YOLO
        model = YOLO(str(weights))
        if getattr(model, "task", None) != "detect":
            raise ValueError("Prediction requires an object detection model")
        names = class_names(getattr(model, "names", None))
    except ImportError as error:
        parser.error(f"Ultralytics/PyTorch dependencies are unavailable; prepare the model environment before prediction: {error}")
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(f"Cannot prepare prediction weights {weights}: {error}")
    try:
        directory = run_directory(Path(plan["project"]), plan["name"])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    partial, final = directory / "predictions.jsonl.part", directory / "predictions.jsonl"
    record = {"started_at": utc_now(), "status": "running", "source": str(source), "source_files": plan["source_files"], "source_fingerprint": plan["source_fingerprint"], "weights": str(weights), "weights_sha256": plan["weights_sha256"], "class_names": names, "confidence": plan["confidence"], "imgsz": plan["imgsz"], "device": device, "images_or_frames": 0}
    allowed = {Path(item["path"]) for item in plan["source_files"]}
    counts = {}
    count = 0
    start = time.perf_counter()
    try:
        write_json(directory / "run.json", record)
        # Line buffering preserves completed rows without collecting the video in RAM.
        with partial.open("w", encoding="utf-8", buffering=1) as handle:
            stream = model.predict(source=str(source), imgsz=plan["imgsz"], conf=plan["confidence"], device=device, save=True, stream=True, project=str(directory.parent), name=directory.name, exist_ok=True)
            try:
                for result in stream:
                    path, row = frame_record(result, names, allowed, counts, count)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    counts[path] = counts.get(path, 0) + 1
                    count += 1
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
        if count == 0:
            raise ValueError("No prediction frames were returned")
        missing = allowed - set(counts)
        if missing:
            raise ValueError("Prediction results are missing for source files: " + ", ".join(str(path) for path in sorted(missing)[:3]))
        if sha256(weights) != plan["weights_sha256"]:
            raise ValueError("Prediction weights changed during the run")
        try:
            _, fingerprint = source_snapshot(source)
        except (ValueError, OSError) as error:
            raise ValueError(f"Prediction source changed or became unreadable: {error}") from error
        if fingerprint != plan["source_fingerprint"]:
            raise ValueError("Prediction source content or membership changed during the run")
        partial.replace(final)
        record.update(status="complete", completed_at=utc_now(), elapsed_seconds=time.perf_counter() - start, images_or_frames=count, predictions_file=str(final))
        write_json(directory / "run.json", record)
    except (Exception, KeyboardInterrupt) as error:
        # If the completion record failed, keep output clearly marked as partial.
        rollback_error = None
        if final.exists() and not partial.exists():
            try:
                final.replace(partial)
            except OSError as move_error:
                rollback_error = str(move_error)
        record.update(status="failed", completed_at=utc_now(), elapsed_seconds=time.perf_counter() - start, images_or_frames=count, predictions_file=None, error=str(error) or type(error).__name__)
        record["partial_predictions_file"] = str(partial) if partial.is_file() else None
        record["unconfirmed_predictions_file"] = str(final) if final.is_file() else None
        if rollback_error:
            record["publication_rollback_error"] = rollback_error
        try:
            write_json(directory / "run.json", record)
        except OSError as record_error:
            parser.error(f"Prediction failed: {error}; cannot save failure record: {record_error}")
        if isinstance(error, KeyboardInterrupt):
            raise
        parser.error(f"Prediction failed: {error}. See {directory / 'run.json'}")
    print(f"Predictions saved to {directory}")


if __name__ == "__main__":
    main()
