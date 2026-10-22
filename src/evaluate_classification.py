"""Evaluate a verified classifier; test requires a completed baseline and an explicit final-test flag."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.preflight_classification import build_transform
from .classification_data import inspect_dataset, read_json
from .classification_model import classification_metrics, completed_training_record, load_classifier, prediction_row
from .common import local_caches, project_path, run_directory, select_device, sha256, utc_now, validate_run_name, write_json
from .train_classification import training_plan


def independent_benchmark(data: Path, reference: Path) -> dict:
    """A new test must be disjoint from ALL old splits, including previously viewed tests."""
    checked = inspect_dataset(reference)
    current = read_json(data / "manifest.json")
    previous = read_json(reference / "manifest.json")
    candidates, old = current["splits"]["test"], [row for rows in previous["splits"].values() for row in rows]
    if current["classes"] != previous["classes"]:
        raise ValueError("Benchmark classes differ from the reference dataset")
    for field in ("id", "group", "sha256", "pixel_sha256"):
        if {row[field] for row in candidates} & {row[field] for row in old}:
            raise ValueError(f"New benchmark overlaps the reference dataset by {field}")
    crop_paths = [Path(manifest["crop_directory"]) / "crops_manifest.json" for manifest in (current, previous)]
    new_ids = {row["id"] for row in candidates}
    new_sources = {row["source_sha256"] for row in read_json(crop_paths[0])["crops"] if row["id"] in new_ids}
    old_sources = {row["source_sha256"] for row in read_json(crop_paths[1])["crops"]}
    if new_sources & old_sources:
        raise ValueError("New benchmark shares original image bytes with the reference dataset")
    return {"reference_dataset_fingerprint": checked["dataset_fingerprint"], "excluded_old_images": len(old),
            "new_test_images": len(candidates), "known_overlap_count": 0,
            "metadata_sha256": {str(path): sha256(path) for path in
                                (reference / "manifest.json", crop_paths[0], crop_paths[1])}}


def build_plan(config: Path, weights: Path, split: str, final_test: bool, name: str,
               benchmark: bool = False) -> tuple[dict, dict]:
    if split not in ("val", "test") or final_test != (split == "test"):
        raise ValueError("Test requires --split test --final-test; use val during model selection")
    validate_run_name(name)
    plan, checked = training_plan(config)
    record = completed_training_record(weights)
    if final_test and record["mode"] != "baseline":
        raise ValueError("Final test requires a baseline, not a smoke run")
    reference = None
    if benchmark:
        if not final_test or record.get("dataset_fingerprint") == checked["dataset_fingerprint"]:
            raise ValueError("External benchmark requires a new dataset and an explicit final test")
        reference = independent_benchmark(Path(plan["data"]), project_path(record["parameters"]["data"]))
        if reference["reference_dataset_fingerprint"] != record["dataset_fingerprint"]:
            raise ValueError("Reference dataset differs from the original training record")
    elif (record.get("config_sha256") != sha256(config)
          or record.get("dataset_fingerprint") != checked["dataset_fingerprint"]):
        raise ValueError("Evaluation config or data differs from the completed training run")
    marker = weights.parent / "final_test.json"
    if benchmark:
        marker = weights.parent / f"benchmark_{checked['dataset_fingerprint']}.json"
    if final_test and (marker.exists() or marker.is_symlink()):
        raise FileExistsError("Final test has already been attempted for this run; read its saved report")
    directory = project_path("results/classification_evaluations") / name
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("Choose a new classifier evaluation name")
    return {"config": str(config), "config_sha256": sha256(config), "weights": str(weights),
            "weights_sha256": sha256(weights), "split": split, "final_test": final_test, "name": name,
            "project": str(directory.parent), "marker": str(marker), "parameters": plan,
            "external_benchmark": reference, "checkpoint_config_sha256": record["config_sha256"],
            "trained_dataset_fingerprint": record["dataset_fingerprint"],
            "training_mode": record["mode"], "training_record_sha256": sha256(weights.parent / "experiment.json")}, checked


def run(config: Path, weights: Path, split: str, final_test: bool, name: str, benchmark: bool = False) -> dict:
    plan, checked = build_plan(config, weights, split, final_test, name, benchmark)
    directory = run_directory(Path(plan["project"]), name)
    result = {"format_version": 1, "task": "classification", "status": "running", "started_at": utc_now(),
              "split": split, "final_test": final_test, "weights_sha256": plan["weights_sha256"],
              "config_sha256": plan["config_sha256"], "dataset_fingerprint": checked["dataset_fingerprint"],
              "training_mode": plan["training_mode"], "training_record_sha256": plan["training_record_sha256"],
              "external_benchmark": plan["external_benchmark"], "trained_dataset_fingerprint": plan["trained_dataset_fingerprint"],
              "decision_rule": "argmax; ties choose lower class ID; no test-set threshold selection",
              "probability_note": "Softmax scores are not calibrated confidence estimates"}
    marker = Path(plan["marker"])
    if final_test:
        # Exclusive creation prevents concurrent/repeated final tests before any model forward.
        with marker.open("x", encoding="utf-8") as handle:
            json.dump({"status": "running", "started_at": result["started_at"], "report": str(directory / "metrics.json"),
                       "weights_sha256": plan["weights_sha256"]}, handle, indent=2)
    write_json(directory / "run.json", result)
    try:
        local_caches()
        import torch
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder

        parameters = plan["parameters"]
        selected = select_device(parameters["device"])
        device = "cpu" if selected == "cpu" else f"cuda:{selected}"
        network, checkpoint = load_classifier(weights, device, classes=parameters["classes"],
                                              dataset_fingerprint=plan["trained_dataset_fingerprint"])
        if checkpoint.get("config_sha256") != plan["checkpoint_config_sha256"] or checkpoint.get("mode") != plan["training_mode"]:
            raise ValueError("Checkpoint config/mode differs from its training record")
        data = Path(parameters["data"])
        dataset = ImageFolder(data / split, transform=build_transform())
        if dataset.class_to_idx != {name: index for index, name in enumerate(parameters["classes"])}:
            raise ValueError("Evaluation class mapping differs")
        loader = DataLoader(dataset, batch_size=parameters["batch"], shuffle=False, num_workers=0)
        matrix = [[0] * len(parameters["classes"]) for _ in parameters["classes"]]
        predictions, loss_sum = [], 0.0
        with torch.inference_mode():
            for images, labels in loader:
                logits = network(images.to(device))
                if logits.shape != (len(images), len(matrix)) or not torch.isfinite(logits).all():
                    raise RuntimeError("Invalid evaluation logits")
                loss_sum += float(torch.nn.functional.cross_entropy(logits, labels.to(device), reduction="sum").item())
                for target, probabilities in zip(labels.tolist(), logits.softmax(1).cpu().tolist()):
                    source = Path(dataset.samples[len(predictions)][0])
                    row = prediction_row(probabilities, parameters["classes"])
                    matrix[target][row["class_id"]] += 1
                    predictions.append({"id": source.stem, "file": source.relative_to(data).as_posix(),
                                        "true_class_id": target, "true_label": parameters["classes"][target], **row})
        if len(predictions) != len(dataset):
            raise RuntimeError("Evaluation missed samples")
        if (inspect_dataset(data)["dataset_fingerprint"] != checked["dataset_fingerprint"]
                or sha256(weights) != plan["weights_sha256"] or sha256(config) != plan["config_sha256"]
                or sha256(weights.parent / "experiment.json") != plan["training_record_sha256"]):
            raise RuntimeError("Evaluation inputs changed")
        if plan["external_benchmark"] and any(sha256(Path(path)) != digest for path, digest in
                                              plan["external_benchmark"]["metadata_sha256"].items()):
            raise RuntimeError("Benchmark reference metadata changed")
        metrics = classification_metrics(matrix, parameters["classes"])
        result.update(status="complete", completed_at=utc_now(), metrics=metrics, loss=loss_sum / len(dataset),
                      predictions=predictions, errors=[row["id"] for row in predictions if row["class_id"] != row["true_class_id"]],
                      all_negative_baseline_accuracy=sum(row["true_class_id"] == 0 for row in predictions) / len(predictions))
        write_json(directory / "metrics.json", result)
    except BaseException as error:
        result.update(status="failed", completed_at=utc_now(), error=str(error))
        raise
    finally:
        write_json(directory / "run.json", result)
        if final_test:
            write_json(marker, {"status": result["status"], "report": str(directory / "metrics.json"),
                                "weights_sha256": plan["weights_sha256"], "completed_at": result.get("completed_at"),
                                "metrics_sha256": sha256(directory / "metrics.json") if result["status"] == "complete" else None})
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--name", required=True)
    parser.add_argument("--benchmark", action="store_true", help="Evaluate on a new test disjoint from all old splits")
    args = parser.parse_args()
    try:
        run(project_path(args.config), project_path(args.weights), args.split, args.final_test, args.name, args.benchmark)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
