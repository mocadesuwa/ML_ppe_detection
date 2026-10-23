"""Plan the reviewed classifier by default; training requires explicit --start."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from scripts.preflight_classification import build_transform, state_hash, validate_runtime_plan
from .classification_data import inspect_dataset, read_json
from .classification_model import (MODEL, PREPROCESSING, classification_metrics, completed_training_record,
                                   configure_trainable_features, feature_blocks_to_train, feature_buffers_hash, load_classifier)
from .common import local_caches, project_path, publish_path, run_directory, select_device, sha256, utc_now, validate_run_name, write_json


def training_plan(config: Path) -> tuple[dict, dict]:
    plan, checked = validate_runtime_plan(config)
    plan.setdefault("mode", "baseline")
    if plan["mode"] not in ("smoke", "baseline"):
        raise ValueError("Classifier mode must be smoke or baseline")
    for key in ("epochs", "patience"):
        if type(plan.get(key)) is not int or plan[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("lr", "weight_decay"):
        value = plan.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or key == "lr" and value == 0:
            raise ValueError(f"Invalid {key}")
    if feature_blocks_to_train(plan):
        for key in ("backbone_lr", "max_grad_norm"):
            value = plan.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Partial fine-tuning requires a positive finite {key}")
        if plan["backbone_lr"] > plan["lr"]:
            raise ValueError("backbone_lr must not exceed the classifier learning rate")
    if plan.get("optimizer") != "AdamW" or type(plan.get("workers")) is not int or plan["workers"] != 0:
        raise ValueError("The first plan uses AdamW and workers=0")
    if plan.get("class_weighting") not in ("inverse_train_frequency", "none"):
        raise ValueError("Class weighting must be inverse_train_frequency or none")
    plan.setdefault("sampling", "shuffle")
    if plan["sampling"] not in ("shuffle", "source_group_balanced", "class_source_group_balanced"):
        raise ValueError("Unknown classifier sampling strategy")
    if plan["sampling"] == "class_source_group_balanced" and plan["class_weighting"] != "none":
        raise ValueError("Class-balanced sampling requires class_weighting=none to avoid double balancing")
    if not isinstance(plan.get("name"), str):
        raise ValueError("A run name is required")
    validate_run_name(plan["name"])
    for key in ("project", "preflight"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise ValueError(f"Explicit {key} path is required")
    counts = checked["counts"]["train"]
    total = sum(counts.values())
    plan["class_weights"] = ([total / (len(plan["classes"]) * counts[name]) for name in plan["classes"]]
                             if plan["class_weighting"] == "inverse_train_frequency" else [1.0] * len(plan["classes"]))
    return plan, checked


def require_preflight(config: Path, plan: dict, checked: dict) -> None:
    report = read_json(project_path(plan["preflight"]))
    if (report.get("status") != "ready_before_training" or report.get("training_started") is not False
            or report.get("gradient_updates") != 0 or report.get("config_sha256") != sha256(config)
            or report.get("initial_weights_sha256") != plan["weights_sha256"]
            or report.get("dataset", {}).get("dataset_fingerprint") != checked["dataset_fingerprint"]):
        raise ValueError("Run an inference-only preflight for this exact config/data/weight version first")
    blocks = feature_blocks_to_train(plan)
    if blocks and (report.get("train_last_blocks") != blocks
                   or report.get("initial_classifier_checkpoint_sha256") != plan["init_checkpoint_sha256"]
                   or report.get("initializer_record_sha256") != sha256(Path(plan["init_checkpoint"]).parent / "experiment.json")):
        raise ValueError("Fine-tuning preflight must match the initializer and trainable scope")


def macro_f1(matrix: list[list[int]]) -> float:
    return classification_metrics(matrix, [str(index) for index in range(len(matrix))])["macro_f1"]


def weighted_loss_totals(batch_loss: float, batch_weight: float, numerator: float, denominator: float) -> tuple[float, float]:
    """Undo each batch's weighted-mean denominator before epoch aggregation."""
    if not math.isfinite(batch_loss) or not math.isfinite(batch_weight) or batch_weight <= 0:
        raise ValueError("Weighted loss must be finite with a positive denominator")
    return numerator + batch_loss * batch_weight, denominator + batch_weight


def source_group_sampling_weights(sample_ids: list[str], rows: list[dict], *, balance_classes: bool = False) -> list[float]:
    by_id = {row["id"]: row["group"] for row in rows}
    if len(by_id) != len(rows) or len(sample_ids) != len(set(sample_ids)) or set(sample_ids) != set(by_id):
        raise ValueError("Sampler must cover exactly the frozen training IDs")
    if balance_classes:
        if not rows or any(not isinstance(row.get("label"), str) or not row["label"].strip() for row in rows):
            raise ValueError("Class-balanced sampling requires explicit training labels")
        labels = {row["id"]: row["label"] for row in rows}
        pairs = Counter((row["label"], row["group"]) for row in rows)
        groups_per_class = Counter(label for label, group in pairs)
        if len(groups_per_class) < 2:
            raise ValueError("Class-balanced sampling requires at least two classes")
        # Each class has mass 1, shared equally across its source groups, then its images.
        return [1.0 / (groups_per_class[labels[identifier]] * pairs[labels[identifier], by_id[identifier]])
                for identifier in sample_ids]
    counts = Counter(by_id.values())
    return [1.0 / counts[by_id[identifier]] for identifier in sample_ids]


def start(config: Path, plan: dict, checked: dict) -> dict:
    require_preflight(config, plan, checked)
    project = project_path(plan["project"])
    data = Path(plan["data"])
    if project == data or data in project.parents:
        raise ValueError("Training evidence must be outside frozen data")
    directory = run_directory(project, plan["name"])
    blocks = feature_blocks_to_train(plan)
    record = {"format_version": 1, "status": "running", "mode": plan["mode"], "started_at": utc_now(), "task": "classification", "parameters": plan,
              "training_strategy": "partial_finetune" if blocks else "head_only",
              "dataset_fingerprint": checked["dataset_fingerprint"], "config_sha256": sha256(config),
              "training_started": True, "test_evaluated": False, "epochs": [], "gradient_updates": 0}
    record["code_sha256"] = {str(path.relative_to(project_path("."))).replace("\\", "/"): sha256(path)
                             for path in (Path(__file__), project_path("src/classification_model.py"),
                                          project_path("scripts/preflight_classification.py"))}
    started = time.perf_counter()
    write_json(directory / "experiment.json", record)
    try:
        local_caches()
        import torch
        import torchvision
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from torchvision.datasets import ImageFolder
        from torchvision.models import mobilenet_v3_small

        torch.manual_seed(plan["seed"])
        device = select_device(plan["device"])
        device = "cpu" if device == "cpu" else f"cuda:{device}"
        record["environment"] = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                                 "device": device, "gpu": None if device == "cpu" else torch.cuda.get_device_name(device),
                                 "cpu_threads": torch.get_num_threads()}
        datasets = {subset: ImageFolder(data / subset, transform=build_transform(subset == "train")) for subset in ("train", "val")}
        mapping = {name: index for index, name in enumerate(plan["classes"])}
        if any(dataset.class_to_idx != mapping for dataset in datasets.values()):
            raise ValueError("ImageFolder class mapping differs from plan")
        generator = torch.Generator().manual_seed(plan["seed"])
        sampler = None
        if plan["sampling"] != "shuffle":
            rows = read_json(data / "manifest.json")["splits"]["train"]
            sample_ids = [Path(path).stem for path, _ in datasets["train"].samples]
            weights = source_group_sampling_weights(sample_ids, rows,
                                                     balance_classes=plan["sampling"] == "class_source_group_balanced")
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
            record["sampling"] = {"strategy": plan["sampling"], "source_groups": len({row["group"] for row in rows}),
                                  "draws_per_epoch": len(weights), "replacement": True}
            labels_by_id = {row["id"]: row["label"] for row in rows}
            record["sampling"]["expected_class_fractions"] = {
                label: sum(weight for identifier, weight in zip(sample_ids, weights) if labels_by_id[identifier] == label) / sum(weights)
                for label in plan["classes"]}
        loaders = {"train": DataLoader(datasets["train"], batch_size=plan["batch"], shuffle=sampler is None,
                                        sampler=sampler, num_workers=0, generator=generator),
                   "val": DataLoader(datasets["val"], batch_size=plan["batch"], shuffle=False, num_workers=0)}
        if blocks:
            initial = Path(plan["init_checkpoint"])
            parent = completed_training_record(initial)
            network, checkpoint = load_classifier(initial, "cpu", classes=plan["classes"], dataset_fingerprint=checked["dataset_fingerprint"])
            if checkpoint.get("config_sha256") != parent["config_sha256"] or checkpoint.get("mode") != "baseline":
                raise ValueError("Fine-tuning checkpoint differs from its completed baseline record")
            record["initializer_checkpoint_sha256"] = sha256(initial)
            record["initializer_record_sha256"] = sha256(initial.parent / "experiment.json")
        else:
            network = mobilenet_v3_small(weights=None)
            network.load_state_dict(torch.load(plan["weights"], map_location="cpu", weights_only=True), strict=True)
            network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, len(mapping))
        configure_trainable_features(network, blocks)
        network.to(device)
        features_before, classifier_before = state_hash(network.features), state_hash(network.classifier)
        frozen_prefix = network.features[:-blocks] if blocks else network.features
        prefix_before, buffers_before = state_hash(frozen_prefix), feature_buffers_hash(network.features)
        tail_before = state_hash(network.features[-blocks:]) if blocks else None
        class_weights = torch.tensor(plan["class_weights"], device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        parameters = [{"params": list(network.classifier.parameters()), "lr": plan["lr"]}]
        if blocks:
            parameters.append({"params": [p for p in network.features.parameters() if p.requires_grad], "lr": plan["backbone_lr"]})
        optimizer = torch.optim.AdamW(parameters, lr=plan["lr"], weight_decay=plan["weight_decay"])
        trainable = [p for p in network.parameters() if p.requires_grad]
        record["optimizer_group_lrs"] = [group["lr"] for group in optimizer.param_groups]
        if device != "cpu":
            torch.cuda.reset_peak_memory_stats(device)
        record["trainable_parameters"] = sum(value.numel() for value in network.parameters() if value.requires_grad)
        best, stale = -1.0, 0
        best_probe = None
        for epoch in range(1, plan["epochs"] + 1):
            network.train()
            network.features.eval()  # Keep every feature BatchNorm statistic fixed, including the trainable tail.
            loss_sum, loss_weight, largest_norm = 0.0, 0.0, 0.0
            for images, labels in loaders["train"]:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(network(images), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                if blocks:
                    norm = torch.nn.utils.clip_grad_norm_(trainable, plan["max_grad_norm"], error_if_nonfinite=True)
                    largest_norm = max(largest_norm, float(norm.item()))
                optimizer.step()
                loss_sum, loss_weight = weighted_loss_totals(float(loss.item()), float(class_weights[labels].sum().item()),
                                                           loss_sum, loss_weight)
                record["gradient_updates"] += 1
            network.eval()
            matrix = [[0] * len(mapping) for _ in mapping]
            val_loss = 0.0
            probe = None
            with torch.inference_mode():
                for images, labels in loaders["val"]:
                    logits = network(images.to(device))
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("Non-finite validation output")
                    val_loss += float(torch.nn.functional.cross_entropy(logits, labels.to(device), reduction="sum").item())
                    predictions = logits.argmax(1).cpu().tolist()
                    if probe is None:
                        probe = (images.clone(), logits.cpu().clone())
                    for target, prediction in zip(labels.tolist(), predictions):
                        matrix[target][prediction] += 1
            score = macro_f1(matrix)
            metrics = classification_metrics(matrix, plan["classes"])
            record["epochs"].append({"epoch": epoch, "train_loss": loss_sum / loss_weight,
                                     "val_loss": val_loss / len(datasets["val"]), "val_accuracy": metrics["accuracy"],
                                     "val_macro_f1": score, "val_confusion_matrix": matrix,
                                     "val_per_class": metrics["per_class"]})
            if blocks:
                record["epochs"][-1]["max_grad_norm_before_clip"] = largest_norm
            if score > best:
                best, stale = score, 0
                best_probe = probe
                checkpoint = {"format_version": 1, "task": "classification", "imgsz": 224, "preprocessing": PREPROCESSING,
                              "state_dict": {key: value.detach().cpu() for key, value in network.state_dict().items()},
                              "classes": plan["classes"], "model": MODEL, "epoch": epoch, "mode": plan["mode"],
                              "dataset_fingerprint": checked["dataset_fingerprint"], "config_sha256": record["config_sha256"]}
                staged = directory / "best.pt.part"
                torch.save(checkpoint, staged)
                publish_path(staged, directory / "best.pt", overwrite=True)
                record.update(best_epoch=epoch, best_val_macro_f1=best)
            else:
                stale += 1
            write_json(directory / "experiment.json", record)
            print(json.dumps({"epoch": epoch, "train_loss": loss_sum / loss_weight,
                              "val_macro_f1": score, "best_epoch": record["best_epoch"]}), flush=True)
            if stale >= plan["patience"]:
                break
        record["frozen_features_unchanged"] = state_hash(network.features) == features_before
        record["classifier_updated"] = state_hash(network.classifier) != classifier_before
        record["frozen_feature_prefix_unchanged"] = state_hash(frozen_prefix) == prefix_before
        record["feature_buffers_unchanged"] = feature_buffers_hash(network.features) == buffers_before
        record["trainable_features_updated"] = blocks > 0 and state_hash(network.features[-blocks:]) != tail_before
        scope_ok = (not record["frozen_features_unchanged"] and record["trainable_features_updated"]
                    if blocks else record["frozen_features_unchanged"])
        if (not scope_ok or not record["classifier_updated"] or not record["frozen_feature_prefix_unchanged"]
                or not record["feature_buffers_unchanged"]):
            raise RuntimeError("Training violated the frozen prefix/BatchNorm or failed to update trainable parameters")
        reloaded, checkpoint = load_classifier(directory / "best.pt", device, classes=plan["classes"],
                                              dataset_fingerprint=checked["dataset_fingerprint"])
        with torch.inference_mode():
            restored = reloaded(best_probe[0].to(device)).cpu()
        record["checkpoint_reload_verified"] = bool(torch.allclose(restored, best_probe[1], rtol=1e-5, atol=1e-6))
        if not record["checkpoint_reload_verified"] or checkpoint["epoch"] != record["best_epoch"]:
            raise RuntimeError("Reloaded best checkpoint differs from its saved validation outputs")
        if (inspect_dataset(data)["dataset_fingerprint"] != checked["dataset_fingerprint"]
                or sha256(Path(plan["weights"])) != plan["weights_sha256"] or sha256(config) != record["config_sha256"]):
            raise RuntimeError("Data, initial weights or config changed during training")
        if blocks and (sha256(Path(plan["init_checkpoint"])) != record["initializer_checkpoint_sha256"]
                       or sha256(Path(plan["init_checkpoint"]).parent / "experiment.json") != record["initializer_record_sha256"]):
            raise RuntimeError("Fine-tuning initializer changed during training")
        record.update(status="complete", completed_at=utc_now(), best_weights_sha256=sha256(directory / "best.pt"))
    except BaseException as error:
        record.update(status="failed", completed_at=utc_now(), error=str(error))
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        if "device" in locals() and device != "cpu" and "torch" in locals():
            record["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
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
