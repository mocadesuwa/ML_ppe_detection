"""Evaluate saved weights, reporting overall and per-class detection metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .check_dataset import validate
from .common import dataset_config, local_caches, project_path, require_review, run_directory, select_device, sha256, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on val by default; use --split test --final-test for final reporting")
    parser.add_argument("--weights", default="results/baseline_01/weights/best.pt")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--final-test", action="store_true", help="Indicate that the model and settings have been fixed")
    parser.add_argument("--name")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.split == "test" and not args.final_test:
        parser.error("Final test requires --final-test; use val while selecting settings")
    weights = project_path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Train the model first or specify existing weights: {weights}")
    config = dataset_config(project_path(args.data))
    report, _ = validate(config)
    if not report["ok"]:
        raise ValueError("Dataset checks failed")
    require_review(config)
    experiment_path = weights.parent.parent / "experiment.json"
    if experiment_path.is_file():
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        if experiment["dataset_fingerprint"] != report["dataset_fingerprint"]:
            raise ValueError("Evaluation data differs from the recorded training dataset")
        if args.split == "test" and experiment.get("mode") == "smoke":
            raise ValueError("Smoke-test weights are not a formal baseline for final testing")
    local_caches()
    device = select_device(args.device)
    from ultralytics import YOLO
    import yaml
    name = args.name or f"{args.split}_{weights.parent.parent.name}"
    directory = run_directory(project_path("results/evaluations"), name)
    data_path = directory / "dataset_resolved.yaml"
    data_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    parameters = {"data": str(data_path), "split": args.split, "imgsz": args.imgsz, "batch": args.batch, "device": device, "workers": 0, "plots": True, "project": str(directory.parent), "name": directory.name, "exist_ok": True}
    model = YOLO(str(weights))
    if {int(i): str(n) for i, n in model.names.items()} != config["names"]:
        raise ValueError("Model class names/order do not match the dataset configuration")
    metrics = model.val(**parameters)
    box = metrics.box
    per_class = []
    available = {int(category): index for index, category in enumerate(box.ap_class_index)}
    counts = report["splits"][args.split]["class_instances"]
    for category, name in config["names"].items():
        index = available.get(category)
        per_class.append({"class_id": category, "name": name, "instances": counts[name], "precision": None if index is None else float(box.p[index]), "recall": None if index is None else float(box.r[index]), "mAP50": None if index is None else float(box.ap50[index]), "mAP50_95": None if index is None else float(box.ap[index])})
    result = {"evaluated_at": utc_now(), "split": args.split, "weights": str(weights), "weights_sha256": sha256(weights), "dataset_fingerprint": report["dataset_fingerprint"], "parameters": parameters, "precision_recall_point": "Ultralytics validator's selected operating point; see saved PR/F1 curves. mAP integrates across confidence thresholds.", "overall": {"precision": float(box.mp), "recall": float(box.mr), "mAP50": float(box.map50), "mAP50_95": float(box.map)}, "per_class": per_class, "speed_ms": {key: float(value) for key, value in metrics.speed.items()}}
    write_json(directory / "metrics.json", result)
    print(json.dumps(result["overall"], indent=2))
    print(f"Saved metrics and plots to {directory}")


if __name__ == "__main__":
    main()
