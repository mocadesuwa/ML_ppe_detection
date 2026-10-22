"""Predict wearing status on already-cropped static head images, with no location boxes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.preflight_classification import build_transform
from .classification_model import completed_training_record, load_classifier, prediction_row
from .common import local_caches, project_path, publish_path, run_directory, select_device, sha256, utc_now, write_json
from .predict import source_snapshot


def run(weights: Path, source: Path, name: str, requested_device: str = "0") -> dict:
    record = completed_training_record(weights)
    weights_digest = sha256(weights)
    record_digest = sha256(weights.parent / "experiment.json")
    files, fingerprint = source_snapshot(source)
    directory = run_directory(project_path("results/classification_predictions"), name)
    partial, final = directory / "predictions.jsonl.part", directory / "predictions.jsonl"
    result = {"format_version": 1, "task": "classification", "status": "running", "started_at": utc_now(),
              "weights_sha256": weights_digest, "training_record_sha256": record_digest,
              "dataset_fingerprint": record["dataset_fingerprint"], "training_mode": record["mode"],
              "source_fingerprint": fingerprint, "image_count": 0,
              "input_contract": "One already-cropped head per static image; no automatic head detection",
              "probability_note": "Softmax scores are not calibrated confidence estimates"}
    write_json(directory / "run.json", result)
    try:
        local_caches()
        import torch
        from PIL import Image, ImageOps

        selected = select_device(requested_device)
        device = "cpu" if selected == "cpu" else f"cuda:{selected}"
        network, checkpoint = load_classifier(weights, device, classes=record["parameters"]["classes"],
                                              dataset_fingerprint=record["dataset_fingerprint"])
        if checkpoint.get("config_sha256") != record["config_sha256"] or checkpoint.get("mode") != record["mode"]:
            raise ValueError("Prediction checkpoint config/mode differs from its completed training record")
        transform = build_transform()
        with partial.open("x", encoding="utf-8") as handle, torch.inference_mode():
            for index, item in enumerate(files):
                with Image.open(item["path"]) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                logits = network(transform(image).unsqueeze(0).to(device))
                if logits.shape != (1, len(checkpoint["classes"])) or not torch.isfinite(logits).all():
                    raise RuntimeError("Invalid classification output")
                row = {"source": item["path"], "source_sha256": item["sha256"], "image_index": index,
                       "image_size": {"width": image.width, "height": image.height},
                       **prediction_row(logits.softmax(1).cpu().tolist()[0], checkpoint["classes"])}
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                result["image_count"] += 1
        if (source_snapshot(source)[1] != fingerprint or sha256(weights) != weights_digest
                or sha256(weights.parent / "experiment.json") != record_digest):
            raise RuntimeError("Prediction sources, weights or training record changed")
        publish_path(partial, final)
        result.update(status="complete", completed_at=utc_now(), predictions_file=str(final),
                      predictions_sha256=sha256(final), classes=checkpoint["classes"], device=device)
    except BaseException as error:
        result.update(status="failed", completed_at=utc_now(), error=str(error))
        raise
    finally:
        write_json(directory / "run.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    try:
        run(project_path(args.weights), project_path(args.source), args.name, args.device)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
