"""Run a checked YOLO smoke test or the formal baseline."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from .check_dataset import validate
from .common import ROOT, config_path, dataset_config, load_yaml, local_caches, normalize_device, require_review, run_directory, select_device, utc_now, validate_run_name, write_json


def validate_plan(config: dict) -> None:
    required = ("data", "model", "project", "name", "epochs", "batch", "imgsz", "device")
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    if any(not isinstance(field, str) for field in config):
        raise ValueError("Training parameter names must be strings")
    for field, minimum in (("epochs", 1), ("batch", 1), ("imgsz", 1), ("patience", 0), ("workers", 0), ("seed", 0)):
        if field in config:
            value = config[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
    for field in ("deterministic", "amp", "plots", "save"):
        if field in config and not isinstance(config[field], bool):
            raise ValueError(f"{field} must be a boolean")
    if "cache" in config and not (isinstance(config["cache"], bool) or isinstance(config["cache"], str) and config["cache"] in ("ram", "disk")):
        raise ValueError("cache must be a boolean, ram, or disk")
    if "fraction" in config:
        value = config["fraction"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1 or not math.isfinite(value):
            raise ValueError("fraction must be a finite number in (0, 1]")
    validate_run_name(config["name"])
    config["device"] = normalize_device(config["device"])


def build_plan(args) -> tuple[dict, dict]:
    path = config_path(args.config, "config", ROOT)
    config = load_yaml(path)
    if args.smoke:
        config.update(name="smoke_01", epochs=3, patience=3, fraction=0.1)
    for key in ("name", "epochs", "batch", "device"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    try:
        validate_plan(config)
        paths = {field: config_path(config[field], field, ROOT) for field in ("data", "model", "project")}
        if paths["project"].exists() and not paths["project"].is_dir():
            raise ValueError("project must be a directory path")
    except ValueError as error:
        raise ValueError(f"Invalid training configuration {path}: {error}") from error
    data = dataset_config(paths["data"])
    config.pop("data")
    config["model"] = str(paths["model"])
    config["project"] = str(paths["project"])
    return config, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline; --dry-run prints the plan without loading YOLO")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--smoke", action="store_true", help="3 epochs on a small training fraction; not a final baseline")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", help="auto, cpu, or a CUDA device index such as 0")
    args = parser.parse_args()
    try:
        config, data = build_plan(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({"mode": "smoke" if args.smoke else "baseline", "train": config, "dataset": data}, ensure_ascii=False, indent=2))
        return
    report, _ = validate(data)
    if not report["ok"]:
        raise ValueError("Dataset validation failed: " + "; ".join(report["errors"][:5]))
    require_review(data)
    local_caches()
    device = select_device(str(config.pop("device")))
    from ultralytics import YOLO
    from .environment import inspect_environment
    import yaml
    directory = run_directory(Path(config["project"]), config["name"])
    data_path = directory / "dataset_resolved.yaml"
    data_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(directory / "environment.json", inspect_environment())
    write_json(directory / "data_check.json", report)
    model_path = Path(config.pop("model"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    config.update(data=str(data_path), device=device, exist_ok=True)
    record = {"started_at": utc_now(), "mode": "smoke" if args.smoke else "baseline", "status": "running", "initial_weights": str(model_path), "dataset_fingerprint": report["dataset_fingerprint"], "requested_parameters": config}
    write_json(directory / "experiment.json", record)
    start = time.perf_counter()
    try:
        model = YOLO(str(model_path))
        model.train(**config)
        actual_directory = Path(model.trainer.save_dir)
        best, last = actual_directory / "weights" / "best.pt", actual_directory / "weights" / "last.pt"
        if not best.is_file():
            raise RuntimeError("Training returned without best.pt")
        record.update(status="complete", completed_at=utc_now(), elapsed_seconds=time.perf_counter() - start, best_weights=str(best), last_weights=str(last), actual_parameters=str(actual_directory / "args.yaml"), actual_save_directory=str(actual_directory))
    except Exception as error:
        record.update(status="failed", completed_at=utc_now(), elapsed_seconds=time.perf_counter() - start, error=str(error))
        raise
    finally:
        write_json(directory / "experiment.json", record)
    print(f"Training complete: {best}")


if __name__ == "__main__":
    main()
