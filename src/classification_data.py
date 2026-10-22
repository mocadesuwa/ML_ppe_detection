"""Export reviewed head crops as a frozen ImageFolder dataset; never train."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .check_head_crops import inspect_crops
from .common import SPLITS, project_path, publish_path, sha256, utc_now, validate_run_name, write_json
from .prepare_head_crops import unique_fields
from .split_head_crops import crop_groups


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_fields)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def check_review_attachment(review: dict) -> None:
    if "audit_file" in review or "audit_sha256" in review:
        if not isinstance(review.get("audit_file"), str) or not review["audit_file"].strip():
            raise ValueError("Invalid review audit attachment")
        if sha256(project_path(review["audit_file"])) != review.get("audit_sha256"):
            raise ValueError("Review audit attachment changed")


def source_coverage(review: dict, assignments: dict, actual: dict, classes: list) -> dict:
    domains = review.get("source_domains")
    if domains is None:
        if review.get("require_training_domain_coverage"):
            raise ValueError("Explicit source domains are required")
        return {}
    if (not isinstance(domains, dict) or set(domains) != set(actual)
            or any(not isinstance(value, str) or not value.strip() for value in domains.values())):
        raise ValueError("Source domain review must cover every crop")
    counts = {subset: {} for subset in SPLITS}
    for identifier, subset in assignments.items():
        domain = domains[identifier]
        table = counts[subset].setdefault(domain, Counter())
        table[actual[identifier]["label"]] += 1
    if review.get("require_training_domain_coverage"):
        for domain in set(domains.values()):
            if set(counts["train"].get(domain, {})) != set(classes):
                raise ValueError("Every reviewed source domain must cover every class in training")
    return {subset: {domain: dict(table) for domain, table in tables.items()} for subset, tables in counts.items()}


def reviewed_plan(crops: Path, split_file: Path, review_file: Path) -> tuple[dict, dict]:
    checked = inspect_crops(crops)
    split = read_json(split_file)
    review = read_json(review_file)
    check_review_attachment(review)
    classes = split.get("classes")
    if not isinstance(classes, list) or len(classes) < 2 or classes != sorted(set(classes)):
        raise ValueError("Explicit classes must be unique and alphabetically ordered for ImageFolder")
    for name in classes:
        validate_run_name(name)
    if len({name.casefold() for name in classes}) != len(classes):
        raise ValueError("Class directories must also be unique on Windows")
    if split.get("crop_fingerprint") != checked["crop_fingerprint"]:
        raise ValueError("Split plan does not match these crops")
    if (review.get("format_version") != 1 or review.get("visual_review_completed") is not True
            or review.get("grouping_review_completed") is not True
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
            or review.get("crop_fingerprint") != checked["crop_fingerprint"]):
        raise ValueError("An explicit completed visual/group review for these exact crops is required")
    actual = {item["id"]: item for item in checked["crops"]}
    accepted = review.get("accepted_ids")
    if not isinstance(accepted, list) or len(accepted) != len(actual) or set(accepted) != set(actual):
        raise ValueError("Review must cover every crop exactly once")
    if set(split.get("splits", {})) != set(SPLITS):
        raise ValueError("Exactly train, val and test are required")
    assignments, splits = {}, {}
    for subset in SPLITS:
        splits[subset] = []
        rows = split["splits"][subset]
        if not isinstance(rows, list) or not rows:
            raise ValueError("Every split must be non-empty")
        for row in rows:
            identifier = row.get("id")
            if identifier in assignments or identifier not in actual:
                raise ValueError("Unknown or repeated split crop")
            item = actual[identifier]
            if (row.get("label") != item.get("label") or row.get("output_file") != item["output_file"]
                    or row.get("output_sha256") != item["output_sha256"]
                    or row.get("source_group") != item["group"] or item.get("label") not in classes):
                raise ValueError("Split metadata does not match crop content, label or source group")
            assignments[identifier] = subset
            splits[subset].append({"id": identifier, "label": item["label"], "group": item["group"],
                                   "file": f"{subset}/{item['label']}/{identifier}.png",
                                   "sha256": item["output_sha256"], "pixel_sha256": item["pixel_sha256"]})
        if {row["label"] for row in splits[subset]} != set(classes):
            raise ValueError("Every class must occur in every split")
    if set(assignments) != set(actual):
        raise ValueError("Split plan omits crops")
    crop_list = checked["crops"]
    for group in crop_groups(crop_list):
        if len({assignments[crop_list[index]["id"]] for index in group}) != 1:
            raise ValueError("Related source groups or duplicate pixels cross splits")
    source_coverage(review, assignments, actual, classes)
    core = {"task": "classification", "classes": classes, "crop_fingerprint": checked["crop_fingerprint"],
            "split_plan_sha256": sha256(split_file), "quality_review_sha256": sha256(review_file), "splits": splits}
    return {"format_version": 1, "created_at": utc_now(), **core, "dataset_fingerprint": fingerprint(core),
            "crop_directory": checked["crop_directory"]}, review


def export_dataset(crops: Path, split_file: Path, review_file: Path, output: Path, *, dry_run: bool = False) -> dict:
    crops, split_file, review_file, output = (path.resolve() for path in (crops, split_file, review_file, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a new classification dataset directory")
    if output == crops or crops in output.parents or output in crops.parents:
        raise ValueError("Classification output must be separate from crop inputs")
    manifest, _ = reviewed_plan(crops, split_file, review_file)
    if dry_run:
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".head-crops-", dir=output.parent) as temporary:
        staged = Path(temporary) / "classification"
        for subset, rows in manifest["splits"].items():
            for row in rows:
                target = staged / row["file"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(crops / "images" / f"{row['id']}.png", target)
                if sha256(target) != row["sha256"]:
                    raise ValueError("Crop content changed during export")
        # Copy the supplied review bytes so the original review hash stays meaningful.
        shutil.copyfile(review_file, staged / "quality_review.json")
        if sha256(staged / "quality_review.json") != manifest["quality_review_sha256"]:
            raise ValueError("Review changed during export")
        if sha256(split_file) != manifest["split_plan_sha256"]:
            raise ValueError("Split plan changed during export")
        if inspect_crops(crops)["crop_fingerprint"] != manifest["crop_fingerprint"]:
            raise ValueError("Crop inputs changed during export")
        write_json(staged / "manifest.json", manifest)
        inspect_dataset(staged)
        publish_path(staged, output)
    return manifest


def inspect_dataset(root: Path) -> dict:
    if root.is_symlink():
        raise ValueError("Classification root may not be a symlink")
    root = root.resolve()
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Classification dataset may not contain symlinks")
    manifest = read_json(root / "manifest.json")
    core = {key: manifest[key] for key in ("task", "classes", "crop_fingerprint", "split_plan_sha256", "quality_review_sha256", "splits")}
    if manifest.get("format_version") != 1 or core["task"] != "classification":
        raise ValueError("Unsupported classification manifest")
    if fingerprint(core) != manifest.get("dataset_fingerprint"):
        raise ValueError("Classification manifest fingerprint changed")
    classes = core["classes"]
    if not isinstance(classes, list) or len(classes) < 2 or classes != sorted(set(classes)):
        raise ValueError("Invalid ImageFolder class order")
    for name in classes:
        validate_run_name(name)
    if len({name.casefold() for name in classes}) != len(classes):
        raise ValueError("Class directories must also be unique on Windows")
    review = read_json(root / "quality_review.json")
    check_review_attachment(review)
    if (sha256(root / "quality_review.json") != manifest.get("quality_review_sha256")
            or review.get("visual_review_completed") is not True
            or review.get("grouping_review_completed") is not True
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
            or review.get("crop_fingerprint") != core["crop_fingerprint"]):
        raise ValueError("Classification quality review is missing, changed or stale")
    checked = inspect_crops(Path(manifest["crop_directory"]))
    if checked["crop_fingerprint"] != core["crop_fingerprint"]:
        raise ValueError("Source crops changed after classification export")
    actual = {item["id"]: item for item in checked["crops"]}
    expected_files = {"manifest.json", "quality_review.json"}
    assignments, counts = {}, {}
    if set(core["splits"]) != set(SPLITS):
        raise ValueError("Exactly three splits are required")
    for subset in SPLITS:
        rows = core["splits"][subset]
        counts[subset] = dict(Counter(row["label"] for row in rows))
        if set(counts[subset]) != set(classes):
            raise ValueError("Every split must contain every class")
        for row in rows:
            identifier, label = row["id"], row["label"]
            expected = f"{subset}/{label}/{identifier}.png"
            if identifier in assignments or identifier not in actual or row["file"] != expected:
                raise ValueError("Unknown, repeated or unsafe classification path")
            original = actual[identifier]
            if (label != original.get("label") or row["group"] != original["group"]
                    or row["sha256"] != original["output_sha256"]
                    or row["pixel_sha256"] != original["pixel_sha256"]
                    or sha256(root / expected) != row["sha256"]):
                raise ValueError("Classification content or labels differ from reviewed crops")
            expected_files.add(expected)
            assignments[identifier] = subset
    accepted = review.get("accepted_ids", [])
    if set(assignments) != set(actual) or len(accepted) != len(actual) or set(accepted) != set(actual):
        raise ValueError("Classification export or review omits crops")
    for group in crop_groups(checked["crops"]):
        if len({assignments[checked['crops'][index]['id']] for index in group}) != 1:
            raise ValueError("Source group or duplicate pixels cross splits")
    domains = source_coverage(review, assignments, actual, classes)
    if {path.relative_to(root).as_posix() for path in paths if path.is_file()} != expected_files:
        raise ValueError("Classification directory contains missing or extra files")
    expected_dirs = {f"{subset}/{label}" for subset in SPLITS for label in classes} | set(SPLITS)
    if {path.relative_to(root).as_posix() for path in paths if path.is_dir()} != expected_dirs:
        raise ValueError("Classification directory contains missing or extra directories")
    return {"integrity_passed": True, "data_ready": True, "training_started": False,
            "dataset_fingerprint": manifest["dataset_fingerprint"], "classes": classes,
            "image_count": len(actual), "counts": counts, "source_group_cross_split_count": 0,
            "crop_fingerprint": manifest["crop_fingerprint"], "source_domain_counts": domains}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    for name in ("crops", "splits", "review", "output"):
        export.add_argument(f"--{name}", required=True)
    export.add_argument("--dry-run", action="store_true")
    check = commands.add_parser("check")
    check.add_argument("--data", required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            result = inspect_dataset(project_path(args.data))
        else:
            result = export_dataset(*(project_path(getattr(args, name)) for name in ("crops", "splits", "review", "output")), dry_run=args.dry_run)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
