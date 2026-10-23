"""Shared classifier checkpoint contract and explicit confusion-matrix metrics."""
from __future__ import annotations

import math
from pathlib import Path

from .classification_data import read_json
from .common import sha256

MODEL = "mobilenet_v3_small"
PREPROCESSING = "head_pad224_imagenet_v1"


def feature_blocks_to_train(plan: dict) -> int:
    if not isinstance(plan, dict):
        raise ValueError("Training scope must be an object")
    blocks = plan.get("train_last_blocks", 0)
    if type(blocks) is not int or not 0 <= blocks <= 12:
        raise ValueError("train_last_blocks must be an integer from 0 to 12; the feature stem stays frozen")
    if plan.get("freeze_backbone") is not (blocks == 0):
        raise ValueError("freeze_backbone must agree with train_last_blocks")
    return blocks


def configure_trainable_features(network, blocks: int) -> None:
    if type(blocks) is not int or not 0 <= blocks < len(network.features):
        raise ValueError("Keep at least one feature block frozen")
    network.features.requires_grad_(False)
    if blocks:
        network.features[-blocks:].requires_grad_(True)
    network.classifier.requires_grad_(True)
    network.features.eval()  # Autograd remains enabled for the tail; BatchNorm statistics stay fixed.


def feature_buffers_hash(network) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in network.named_buffers():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def classification_metrics(matrix: list[list[int]], classes: list[str]) -> dict:
    size = len(classes)
    if (not size or len(matrix) != size or any(len(row) != size for row in matrix)
            or any(type(value) is not int or value < 0 for row in matrix for value in row)):
        raise ValueError("Confusion matrix must contain nonnegative integer counts for every class")
    total = sum(sum(row) for row in matrix)
    if not total:
        raise ValueError("Cannot evaluate an empty classification set")
    per_class = []
    for index, name in enumerate(classes):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        per_class.append({"class_id": index, "name": name, "support": support,
                          "precision": tp / predicted if predicted else 0.0,
                          "recall": tp / support if support else 0.0,
                          "f1": 2 * tp / (support + predicted) if support + predicted else 0.0})
    return {"image_count": total, "accuracy": sum(matrix[i][i] for i in range(size)) / total,
            "macro_f1": sum(row["f1"] for row in per_class) / size, "per_class": per_class,
            "confusion_matrix": matrix, "confusion_matrix_axes": "rows=true, columns=predicted",
            "zero_division": 0}


def validate_checkpoint_metadata(checkpoint: dict, classes: list[str] | None = None) -> list[str]:
    names = checkpoint.get("classes")
    if (checkpoint.get("format_version") != 1 or checkpoint.get("task") != "classification"
            or checkpoint.get("model") != MODEL or checkpoint.get("imgsz") != 224
            or checkpoint.get("preprocessing") != PREPROCESSING
            or not isinstance(names, list) or len(names) != 2
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or names != sorted(set(names)) or classes is not None and names != classes):
        raise ValueError("Classifier checkpoint task, class order or preprocessing differs from the contract")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Classifier checkpoint has no state dictionary")
    return names


def completed_training_record(weights: Path) -> dict:
    record = read_json(weights.parent / "experiment.json")
    strategy = record.get("training_strategy", "head_only")
    if strategy == "head_only":
        parameters = record.get("parameters", {})
        scope_verified = (record.get("frozen_features_unchanged") is True
                          and isinstance(parameters, dict) and parameters.get("train_last_blocks", 0) == 0
                          and parameters.get("freeze_backbone", True) is True)
    elif strategy == "partial_finetune":
        blocks = feature_blocks_to_train(record.get("parameters", {}))
        scope_verified = (blocks > 0 and record.get("frozen_features_unchanged") is False
                          and all(record.get(key) is True for key in ("frozen_feature_prefix_unchanged",
                                  "trainable_features_updated", "feature_buffers_unchanged")))
    else:
        scope_verified = False
    if (record.get("task") != "classification" or record.get("status") != "complete"
            or record.get("mode") not in ("smoke", "baseline") or weights.name != "best.pt"
            or record.get("best_weights_sha256") != sha256(weights)
            or not record.get("checkpoint_reload_verified") or not scope_verified
            or not record.get("classifier_updated")):
        raise ValueError("Weights must match the verified best checkpoint of a completed classifier run")
    return record


def load_classifier(weights: Path, device: str, *, classes: list[str] | None = None,
                    dataset_fingerprint: str | None = None):
    import torch
    from torchvision.models import mobilenet_v3_small

    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Classifier checkpoint must be a dictionary")
    names = validate_checkpoint_metadata(checkpoint, classes)
    if dataset_fingerprint is not None and checkpoint.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Classifier checkpoint dataset fingerprint differs")
    network = mobilenet_v3_small(weights=None)
    network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, len(names))
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in network.state_dict().values()):
        raise ValueError("Classifier checkpoint contains non-finite parameters")
    return network.to(device).eval(), checkpoint


def prediction_row(probabilities: list[float], classes: list[str]) -> dict:
    if (len(probabilities) != len(classes) or not probabilities
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities)
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)):
        raise ValueError("Invalid classifier probabilities")
    index = max(range(len(classes)), key=probabilities.__getitem__)
    return {"class_id": index, "label": classes[index], "score": probabilities[index],
            "probabilities": {name: value for name, value in zip(classes, probabilities)}}
