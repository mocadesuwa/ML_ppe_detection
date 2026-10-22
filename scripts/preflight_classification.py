"""Decode a reviewed classification batch and run eval-only forward; never train."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classification_data import inspect_dataset
from src.common import load_yaml, local_caches, project_path, select_device, sha256, utc_now, write_json
from src.environment import inspect_environment


def validate_runtime_plan(path: Path) -> tuple[dict, dict]:
    plan = load_yaml(path)
    if plan.get("task") != "classification" or plan.get("model") != "mobilenet_v3_small":
        raise ValueError("This preflight requires the explicit MobileNetV3 Small classification plan")
    for name in ("data", "weights", "weights_sha256", "dataset_fingerprint", "classes", "device", "imgsz", "batch", "seed"):
        if name not in plan:
            raise ValueError(f"Missing classification plan field: {name}")
    for name in ("imgsz", "batch", "seed"):
        if type(plan[name]) is not int or plan[name] < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid integer parameter: {name}")
    if plan["imgsz"] != 224 or plan.get("freeze_backbone") is not True:
        raise ValueError("The reviewed plan uses 224px inputs and a frozen feature backbone")
    plan["data"] = str(project_path(plan["data"]))
    plan["weights"] = str(project_path(plan["weights"]))
    checked = inspect_dataset(Path(plan["data"]))
    if checked["dataset_fingerprint"] != plan["dataset_fingerprint"] or checked["classes"] != plan["classes"]:
        raise ValueError("Dataset fingerprint or class mapping differs from the reviewed plan")
    if plan["batch"] < len(plan["classes"]):
        raise ValueError("Preflight batch must cover every configured class")
    if not Path(plan["weights"]).is_file():
        raise FileNotFoundError("Provide recorded local classification weights; preflight never downloads")
    if sha256(Path(plan["weights"])) != plan["weights_sha256"]:
        raise ValueError("Initial classification weights changed")
    return plan, checked


def build_transform(training: bool = False):
    from PIL import Image, ImageOps
    from torchvision import transforms

    operations = []
    if training:
        operations.extend([transforms.RandomHorizontalFlip(.5),
                           transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.05)])
    # Preserve the complete reviewed head; no center crop or random eye clipping.
    operations.extend([transforms.Lambda(lambda image: ImageOps.pad(image, (224, 224),
                       method=Image.Resampling.BILINEAR, color=(124, 116, 104))),
                       transforms.ToTensor(),
                       transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    return transforms.Compose(operations)


def state_hash(network) -> str:
    digest = hashlib.sha256()
    for name, tensor in network.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run(config: Path, output: Path) -> dict:
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a new preflight evidence directory")
    config_digest = sha256(config)
    plan, checked = validate_runtime_plan(config)
    data = Path(plan["data"])
    if output == data or data in output.parents:
        raise ValueError("Preflight output must be outside frozen data")
    local_caches()
    environment = inspect_environment()
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.models import mobilenet_v3_small

    device = select_device(plan["device"])
    torch_device = "cpu" if device == "cpu" else f"cuda:{device}"
    torch.manual_seed(plan["seed"])
    datasets = {subset: ImageFolder(data / subset, transform=build_transform()) for subset in ("train", "val", "test")}
    mapping = {name: index for index, name in enumerate(plan["classes"])}
    if any(dataset.class_to_idx != mapping for dataset in datasets.values()):
        raise ValueError("ImageFolder class IDs differ across splits or plan")
    targets = datasets["train"].targets
    first_per_class = [targets.index(index) for index in range(len(mapping))]
    indices = first_per_class + [index for index in range(len(targets)) if index not in first_per_class]
    loader = DataLoader(datasets["train"], batch_size=plan["batch"], sampler=indices, num_workers=0)
    images, labels = next(iter(loader))
    if tuple(images.shape[1:]) != (3, 224, 224) or not torch.isfinite(images).all():
        raise ValueError("Invalid decoded classification batch")
    augmented = build_transform(training=True)(datasets["train"].loader(datasets["train"].samples[0][0]))
    if augmented.shape != (3, 224, 224) or not torch.isfinite(augmented).all():
        raise ValueError("Invalid training-only transform output")
    torch.manual_seed(plan["seed"])
    network = mobilenet_v3_small(weights=None)
    network.load_state_dict(torch.load(plan["weights"], map_location="cpu", weights_only=True), strict=True)
    network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, len(mapping))
    for parameter in network.features.parameters():
        parameter.requires_grad_(False)
    network.to(torch_device).eval()
    initial_state = state_hash(network)
    if torch_device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        logits = network(images.to(torch_device))
    if logits.shape != (len(images), len(mapping)) or not torch.isfinite(logits).all():
        raise RuntimeError("Invalid classification forward output")
    if network.training or any(parameter.grad is not None for parameter in network.parameters()):
        raise RuntimeError("Preflight must not enable training or create gradients")
    if state_hash(network) != initial_state:
        raise RuntimeError("Model parameters or buffers changed during preflight")
    after = inspect_dataset(data)
    if (after["dataset_fingerprint"] != checked["dataset_fingerprint"]
            or sha256(Path(plan["weights"])) != plan["weights_sha256"] or sha256(config) != config_digest):
        raise RuntimeError("Data, weights or configuration changed during preflight")
    report = {"checked_at": utc_now(), "status": "ready_before_training", "training_started": False,
              "gradient_updates": 0, "optimizer_constructed": False, "parameters_and_buffers_unchanged": True,
              "dataset": checked, "environment": environment, "class_to_idx": mapping,
              "config_sha256": config_digest, "initial_weights_sha256": plan["weights_sha256"],
              "input_shape": list(images.shape), "output_shape": list(logits.shape), "device": torch_device,
              "loaded_train_class_ids": labels.tolist(), "training_transform_checked": True,
              "pretrained_role": "ImageNet initialization with a new, untrained two-class head",
              "validation_or_test_model_forward": False,
              "inference_peak_allocated_mib": None if torch_device == "cpu" else torch.cuda.max_memory_allocated(device) / 1024**2,
              "limitations": "No backward, optimizer, fitting, validation selection or test metrics; forward memory does not establish training memory."}
    write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        run(project_path(args.config), project_path(args.output))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
