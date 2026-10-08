"""Check real data loading and an inference-only GPU forward; never train."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.check_dataset import validate
from src.common import load_yaml, local_caches, project_path, require_review, select_device, sha256, utc_now, write_json
from src.environment import inspect_environment
from src.train import build_plan


def run(args) -> dict:
    plan, data = build_plan(args)
    checked, _ = validate(data)
    if not checked["ok"]:
        raise ValueError("Resolve dataset errors before runtime preflight")
    require_review(data)
    weights, destination = Path(plan["model"]), project_path(args.output)
    if not weights.is_file():
        raise FileNotFoundError("Download and record local initial weights before preflight")
    if destination.exists():
        raise FileExistsError("Preflight output exists; choose a new directory")
    if Path(data["path"]) in destination.parents or destination == Path(data["path"]):
        raise ValueError("Preflight evidence must be outside the frozen dataset")
    initial_hash = sha256(weights)
    local_caches()
    environment = inspect_environment()
    device = select_device(str(plan["device"]))
    if device == "cpu" or not environment.get("gpu_operation_ok"):
        raise RuntimeError("This preflight requires the prepared CUDA environment")
    import torch
    from torchvision.ops import nms
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset

    # Ultralytics validates the eventual configuration without constructing a trainer.
    data_file = project_path(load_yaml(project_path(args.config))["data"])
    overrides = {**plan, "device": device, "data": str(data_file)}
    hyp = get_cfg(overrides=overrides)
    model = YOLO(str(weights))
    if model.task != "detect":
        raise ValueError("Initial model must be a detection model")
    network = model.model.to(f"cuda:{device}").eval()
    dataset = YOLODataset(img_path=data["train"], data=data, imgsz=plan["imgsz"], batch_size=plan["batch"],
                          augment=False, hyp=hyp, rect=False, cache=False, task="detect", prefix="preflight: ")
    size = min(plan["batch"], len(dataset))
    if not size:
        raise ValueError("Training split is empty")
    batch = dataset.collate_fn([dataset[i] for i in range(size)])
    images = batch["img"].to(f"cuda:{device}").float() / 255
    if images.shape != (size, 3, plan["imgsz"], plan["imgsz"]) or not torch.isfinite(images).all():
        raise ValueError("Unexpected decoded image tensor")
    for key in ("cls", "bboxes"):
        if not torch.isfinite(batch[key]).all():
            raise ValueError(f"Non-finite loaded labels: {key}")
    if batch["cls"].numel() and not ((batch["cls"] >= 0) & (batch["cls"] < len(data["names"]))).all():
        raise ValueError("Loaded class IDs exceed the configured task")
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), torch.autocast("cuda", enabled=plan.get("amp", False)):
        output = network(images)
        prediction = output[0] if isinstance(output, tuple) else output
        if not isinstance(prediction, torch.Tensor) or prediction.shape[0] != size or not torch.isfinite(prediction).all():
            raise RuntimeError("GPU forward produced invalid predictions")
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=f"cuda:{device}")
        scores = torch.tensor([.9, .8], device=f"cuda:{device}")
        if nms(boxes, scores, .5).tolist() != [0]:
            raise RuntimeError("CUDA torchvision NMS failed")
    torch.cuda.synchronize(device)
    if network.training or any(p.grad is not None for p in network.parameters()):
        raise RuntimeError("Preflight must not create gradients or enable training")
    after, _ = validate(data)
    require_review(data)
    if not after["ok"] or after["dataset_fingerprint"] != checked["dataset_fingerprint"] or sha256(weights) != initial_hash:
        raise RuntimeError("Data or initial weight content changed during preflight")
    report = {"checked_at": utc_now(), "status": "ready_before_training", "training_started": False,
              "environment": environment, "parameters": plan, "data_config": str(data_file), "dataset_fingerprint": checked["dataset_fingerprint"],
              "weights_sha256": initial_hash, "initial_model_task": model.task, "initial_pretrained_classes": len(model.names),
              "configured_classes": data["names"], "loaded_training_images": [str(p) for p in batch["im_file"]],
              "input_shape": list(images.shape), "prediction_shape": list(prediction.shape),
              "cuda_nms_ok": True, "gradient_updates": 0, "inference_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
              "content_unchanged": True, "limitations": "Forward only, no augmentation/backprop/optimizer/trainer. Training memory, smoke, checkpoint saving and task metrics remain untested."}
    write_json(destination / "report.json", report)
    print(f"Ready before training; no gradient updates. Evidence: {destination / 'report.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output", default="results/preflight_01")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    # Keep build_plan's optional overrides explicit; no smoke or real training mode.
    args.smoke, args.name, args.epochs = False, None, None
    run(args)
