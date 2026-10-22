"""Plan the reviewed classifier by default; training requires explicit --start."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.preflight_classification import build_transform, validate_runtime_plan
from .classification_data import inspect_dataset, read_json
from .common import local_caches, project_path, publish_path, run_directory, select_device, sha256, utc_now, validate_run_name, write_json


def training_plan(config: Path) -> tuple[dict, dict]:
    plan, checked = validate_runtime_plan(config)
    for key in ("epochs", "patience"):
        if type(plan.get(key)) is not int or plan[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("lr", "weight_decay"):
        value = plan.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or key == "lr" and value == 0:
            raise ValueError(f"Invalid {key}")
    if plan.get("optimizer") != "AdamW" or type(plan.get("workers")) is not int or plan["workers"] != 0:
        raise ValueError("The first plan uses AdamW and workers=0")
    if plan.get("class_weighting") != "inverse_train_frequency":
        raise ValueError("Class weights must use training counts only")
    if not isinstance(plan.get("name"), str):
        raise ValueError("A run name is required")
    validate_run_name(plan["name"])
    for key in ("project", "preflight"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise ValueError(f"Explicit {key} path is required")
    counts = checked["counts"]["train"]
    total = sum(counts.values())
    plan["class_weights"] = [total / (len(plan["classes"]) * counts[name]) for name in plan["classes"]]
    return plan, checked


def require_preflight(config: Path, plan: dict, checked: dict) -> None:
    report = read_json(project_path(plan["preflight"]))
    if (report.get("status") != "ready_before_training" or report.get("training_started") is not False
            or report.get("gradient_updates") != 0 or report.get("config_sha256") != sha256(config)
            or report.get("initial_weights_sha256") != plan["weights_sha256"]
            or report.get("dataset", {}).get("dataset_fingerprint") != checked["dataset_fingerprint"]):
        raise ValueError("Run an inference-only preflight for this exact config/data/weight version first")


def macro_f1(matrix: list[list[int]]) -> float:
    values = []
    for index in range(len(matrix)):
        tp = matrix[index][index]
        denominator = sum(matrix[index]) + sum(row[index] for row in matrix)
        values.append(2 * tp / denominator if denominator else 0.0)
    return sum(values) / len(values)


def start(config: Path, plan: dict, checked: dict) -> dict:
    require_preflight(config, plan, checked)
    project = project_path(plan["project"])
    data = Path(plan["data"])
    if project == data or data in project.parents:
        raise ValueError("Training evidence must be outside frozen data")
    directory = run_directory(project, plan["name"])
    record = {"status": "running", "started_at": utc_now(), "task": "classification", "parameters": plan,
              "dataset_fingerprint": checked["dataset_fingerprint"], "config_sha256": sha256(config),
              "training_started": True, "test_evaluated": False, "epochs": [], "gradient_updates": 0}
    write_json(directory / "experiment.json", record)
    try:
        local_caches()
        import torch
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder
        from torchvision.models import mobilenet_v3_small

        torch.manual_seed(plan["seed"])
        device = select_device(plan["device"])
        device = "cpu" if device == "cpu" else f"cuda:{device}"
        datasets = {subset: ImageFolder(data / subset, transform=build_transform(subset == "train")) for subset in ("train", "val")}
        mapping = {name: index for index, name in enumerate(plan["classes"])}
        if any(dataset.class_to_idx != mapping for dataset in datasets.values()):
            raise ValueError("ImageFolder class mapping differs from plan")
        generator = torch.Generator().manual_seed(plan["seed"])
        loaders = {subset: DataLoader(dataset, batch_size=plan["batch"], shuffle=subset == "train",
                                     num_workers=0, generator=generator if subset == "train" else None)
                   for subset, dataset in datasets.items()}
        network = mobilenet_v3_small(weights=None)
        network.load_state_dict(torch.load(plan["weights"], map_location="cpu", weights_only=True), strict=True)
        network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, len(mapping))
        network.features.requires_grad_(False)
        network.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(plan["class_weights"], device=device))
        optimizer = torch.optim.AdamW(network.classifier.parameters(), lr=plan["lr"], weight_decay=plan["weight_decay"])
        best, stale = -1.0, 0
        for epoch in range(1, plan["epochs"] + 1):
            network.train()
            network.features.eval()  # Frozen feature BatchNorm buffers must also stay frozen.
            loss_sum = 0.0
            for images, labels in loaders["train"]:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(network(images), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * len(images)
                record["gradient_updates"] += 1
            network.eval()
            matrix = [[0] * len(mapping) for _ in mapping]
            with torch.inference_mode():
                for images, labels in loaders["val"]:
                    predictions = network(images.to(device)).argmax(1).cpu().tolist()
                    for target, prediction in zip(labels.tolist(), predictions):
                        matrix[target][prediction] += 1
            score = macro_f1(matrix)
            record["epochs"].append({"epoch": epoch, "train_loss": loss_sum / len(datasets["train"]),
                                     "val_macro_f1": score, "val_confusion_matrix": matrix})
            if score > best:
                best, stale = score, 0
                checkpoint = {"state_dict": {key: value.detach().cpu() for key, value in network.state_dict().items()},
                              "classes": plan["classes"], "model": plan["model"], "epoch": epoch,
                              "dataset_fingerprint": checked["dataset_fingerprint"], "config_sha256": record["config_sha256"]}
                staged = directory / "best.pt.part"
                torch.save(checkpoint, staged)
                publish_path(staged, directory / "best.pt", overwrite=True)
                record.update(best_epoch=epoch, best_val_macro_f1=best)
            else:
                stale += 1
            write_json(directory / "experiment.json", record)
            if stale >= plan["patience"]:
                break
        if (inspect_dataset(data)["dataset_fingerprint"] != checked["dataset_fingerprint"]
                or sha256(Path(plan["weights"])) != plan["weights_sha256"] or sha256(config) != record["config_sha256"]):
            raise RuntimeError("Data, initial weights or config changed during training")
        record.update(status="complete", completed_at=utc_now(), best_weights_sha256=sha256(directory / "best.pt"))
    except BaseException as error:
        record.update(status="failed", completed_at=utc_now(), error=str(error))
        raise
    finally:
        write_json(directory / "experiment.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--start", action="store_true", help="Explicitly start gradient training; omit to plan only")
    modes.add_argument("--dry-run", action="store_true", help="Check and print the plan, with no model or optimizer")
    args = parser.parse_args()
    try:
        config = project_path(args.config)
        plan, checked = training_plan(config)
        if args.start:
            result = start(config, plan, checked)
        else:
            result = {"status": "planned", "training_started": False, "requires_explicit_start": True,
                      "parameters": plan, "dataset": checked}
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
