"""Save boxes, names, confidence scores and images for local inputs."""
from __future__ import annotations

import argparse

from .common import local_caches, project_path, run_directory, select_device, sha256, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict on a local image, image directory or video")
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", default="results/baseline_01/weights/best.pt")
    parser.add_argument("--name", default="baseline_examples")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between zero and one")
    source, weights = project_path(args.source), project_path(args.weights)
    if not source.exists() or not weights.is_file():
        raise FileNotFoundError("Source and trained weights must both exist")
    local_caches()
    device = select_device(args.device)
    from ultralytics import YOLO
    directory = run_directory(project_path("results/predictions"), args.name)
    model = YOLO(str(weights))
    count = 0
    # JSONL avoids collecting an entire video in RAM.
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        import json
        for frame, result in enumerate(model.predict(source=str(source), imgsz=args.imgsz, conf=args.conf, device=device, save=True, stream=True, project=str(directory.parent), name=directory.name, exist_ok=True)):
            detections = []
            if result.boxes is not None:
                for box, category, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                    detections.append({"class_id": int(category), "name": result.names[int(category)], "confidence": confidence, "xyxy_pixels": box})
            handle.write(json.dumps({"source": str(result.path), "frame_index": frame, "detections": detections}, ensure_ascii=False) + "\n")
            count += 1
    write_json(directory / "run.json", {"completed_at": utc_now(), "source": str(source), "weights": str(weights), "weights_sha256": sha256(weights), "confidence": args.conf, "imgsz": args.imgsz, "device": device, "images_or_frames": count})
    print(f"Predictions saved to {directory}")


if __name__ == "__main__":
    main()
