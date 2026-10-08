"""Evaluate saved weights, reporting overall and per-class detection metrics."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .check_dataset import validate
from .common import config_path, dataset_config, dataset_fingerprint, local_caches, normalize_device, project_path, require_review, run_directory, select_device, sha256, utc_now, validate_run_name, write_json


def training_record(weights: Path, fingerprint: str, split: str) -> dict | None:
    path = weights.parent.parent / "experiment.json"
    if not path.exists() and not path.is_symlink():
        if split == "test":
            raise ValueError(f"Final test requires a completed baseline record: {path}")
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read training record {path}: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"Training record must be a JSON object: {path}")
    digest = record.get("dataset_fingerprint")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"Invalid dataset_fingerprint in training record: {path}")
    if record.get("mode") not in ("baseline", "smoke") or record.get("status") not in ("running", "complete", "failed"):
        raise ValueError(f"Invalid mode/status in training record: {path}")
    checkpoints = []
    for field in ("best_weights", "last_weights"):
        if field in record:
            try:
                checkpoints.append(config_path(record[field], field, project_path(".")))
            except ValueError as error:
                raise ValueError(f"Invalid checkpoint in training record {path}: {error}") from error
    if digest != fingerprint:
        raise ValueError(f"Evaluation dataset differs from the recorded training dataset: {path}")
    if split == "test":
        if record["mode"] != "baseline" or record["status"] != "complete":
            raise ValueError(f"Final test requires a completed baseline, not a smoke/partial run: {path}")
        if weights not in checkpoints:
            raise ValueError(f"Final test weights must match a recorded best/last checkpoint: {path}")
    return {"path": str(path), "mode": record["mode"], "status": record["status"], "dataset_fingerprint": digest}


def build_plan(args) -> tuple[dict, dict, dict]:
    """Check inputs and provenance before importing model dependencies or writing outputs."""
    for field in ("imgsz", "batch"):
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if args.split == "test" and not args.final_test:
        raise ValueError("Final test requires --final-test; use val while selecting settings")
    if args.final_test and args.split != "test":
        raise ValueError("--final-test requires --split test")
    device = normalize_device(args.device)
    weights = config_path(args.weights, "weights", project_path("."))
    data_path = config_path(args.data, "data", project_path("."))
    name = args.name if args.name is not None else f"{args.split}_{weights.parent.parent.name}"
    validate_run_name(name)
    project = project_path("results/evaluations")
    for parent in (project, *project.parents):
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"Evaluation output parent must be a directory: {parent}")
    directory = project / name
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Evaluation already exists; choose a new name: {directory}")
    if not weights.is_file():
        raise FileNotFoundError(f"Train the model first or specify existing weights: {weights}")
    config = dataset_config(data_path)
    report, _ = validate(config)
    if not report["ok"]:
        raise ValueError("Dataset checks failed: " + "; ".join(report["errors"][:5]))
    require_review(config)
    record = training_record(weights, report["dataset_fingerprint"], args.split)
    plan = {"weights": str(weights), "weights_sha256": sha256(weights), "split": args.split, "final_test": args.final_test, "name": name, "project": str(project), "device": device, "imgsz": args.imgsz, "batch": args.batch, "training_record": record}
    return plan, config, report


def check_model_names(names: object, expected: dict) -> None:
    message = "Model class names/order do not match the dataset configuration"
    if not isinstance(names, (dict, list)):
        raise ValueError(message)
    normalized = {}
    for key, name in (enumerate(names) if isinstance(names, list) else names.items()):
        if isinstance(key, bool) or not (isinstance(key, int) or isinstance(key, str) and re.fullmatch(r"[0-9]+", key)):
            raise ValueError(message)
        index = int(key)
        if index in normalized or not isinstance(name, str):
            raise ValueError(message)
        normalized[index] = name
    if normalized != expected:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on val by default; use --split test --final-test for final reporting")
    parser.add_argument("--weights", default="results/baseline_01/weights/best.pt")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--final-test", action="store_true", help="Requires --split test, a completed baseline record, and a recorded best/last checkpoint")
    parser.add_argument("--name")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a non-negative CUDA device index")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    try:
        plan, config, report = build_plan(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    weights = Path(plan["weights"])
    try:
        device = select_device(plan["device"])
        local_caches()
        from ultralytics import YOLO
        model = YOLO(str(weights))
        check_model_names(getattr(model, "names", None), config["names"])
    except ImportError as error:
        parser.error(f"Ultralytics/PyTorch dependencies are unavailable; prepare the model environment before evaluation: {error}")
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(f"Cannot prepare evaluation weights {weights}: {error}")
    import yaml
    try:
        directory = run_directory(Path(plan["project"]), plan["name"])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    data_path = directory / "dataset_resolved.yaml"
    data_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    parameters = {"data": str(data_path), "split": args.split, "imgsz": args.imgsz, "batch": args.batch, "device": device, "workers": 0, "plots": True, "project": str(directory.parent), "name": directory.name, "exist_ok": True}
    metrics = model.val(**parameters)
    try:
        if sha256(weights) != plan["weights_sha256"]:
            raise ValueError("Evaluation weights changed during the run; metrics were not published")
        if dataset_fingerprint(config) != report["dataset_fingerprint"]:
            raise ValueError("Evaluation dataset changed during the run; metrics were not published")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    box = metrics.box
    per_class = []
    available = {int(category): index for index, category in enumerate(box.ap_class_index)}
    counts = report["splits"][args.split]["class_instances"]
    for category, name in config["names"].items():
        index = available.get(category)
        per_class.append({"class_id": category, "name": name, "instances": counts[name], "precision": None if index is None else float(box.p[index]), "recall": None if index is None else float(box.r[index]), "mAP50": None if index is None else float(box.ap50[index]), "mAP50_95": None if index is None else float(box.ap[index])})
    result = {"evaluated_at": utc_now(), "split": args.split, "final_test": plan["final_test"], "weights": str(weights), "weights_sha256": plan["weights_sha256"], "dataset_fingerprint": report["dataset_fingerprint"], "training_record": plan["training_record"], "parameters": parameters, "precision_recall_point": "Ultralytics validator's selected operating point; see saved PR/F1 curves. mAP integrates across confidence thresholds.", "overall": {"precision": float(box.mp), "recall": float(box.mr), "mAP50": float(box.map50), "mAP50_95": float(box.map)}, "per_class": per_class, "speed_ms": {key: float(value) for key, value in metrics.speed.items()}}
    write_json(directory / "metrics.json", result)
    print(json.dumps(result["overall"], indent=2))
    print(f"Saved metrics and plots to {directory}")


if __name__ == "__main__":
    main()
