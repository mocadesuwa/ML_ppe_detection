"""Check labels, cross-split duplicates and generate inspection sheets."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .common import IMAGE_SUFFIXES, SPLITS, dataset_config, dataset_fingerprint, project_path, sha256, utc_now, write_json


def read_labels(path: Path, class_count: int) -> list[list[float]]:
    if isinstance(class_count, bool) or not isinstance(class_count, int) or class_count <= 0:
        raise ValueError("Class count must be a positive integer")
    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Line {line_number}: expected five fields")
        try:
            exact = [Decimal(part) for part in parts]
            values = list(map(float, parts))
        except (InvalidOperation, ValueError, OverflowError) as error:
            raise ValueError(f"Line {line_number}: expected numeric fields") from error
        category, x, y, width, height = values
        if not all(v.is_finite() for v in exact) or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Line {line_number}: non-finite value")
        # Check the original numeric tokens before float rounding can hide errors.
        if exact[0] != exact[0].to_integral_value() or not 0 <= exact[0] < class_count:
            raise ValueError(f"Line {line_number}: invalid class ID")
        if not (0 < exact[1] < 1 and 0 < exact[2] < 1 and 0 < exact[3] <= 1 and 0 < exact[4] <= 1
                and 0 < x < 1 and 0 < y < 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Line {line_number}: invalid normalized box")
        # The converter writes eight decimals: a corner can round by up to 7.5e-9.
        tolerance = 1e-8
        if not (x - width / 2 >= -tolerance and x + width / 2 <= 1 + tolerance and y - height / 2 >= -tolerance and y + height / 2 <= 1 + tolerance):
            raise ValueError(f"Line {line_number}: invalid normalized box")
        labels.append(values)
    return labels


def read_manifest(path: Path, class_count: int, errors: list[str]) -> dict | None:
    """Return usable records while reporting malformed frozen metadata."""
    if not path.exists():
        if any((path.parent / "splits" / f"{split}.txt").exists() for split in SPLITS):
            errors.append("Frozen split lists exist but the manifest is missing")
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        errors.append(f"Cannot read manifest: {error}")
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("splits"), dict) or set(manifest["splits"]) != set(SPLITS):
        errors.append("Manifest must contain train, val and test record lists")
        return None
    version = manifest.get("format_version", 0)
    if isinstance(version, bool) or not isinstance(version, int) or version not in (0, 1):
        errors.append("Unsupported manifest format version")
        return None
    cleaned = dict(manifest, splits={})
    for split in SPLITS:
        records = manifest["splits"][split]
        cleaned["splits"][split] = []
        if not isinstance(records, list):
            errors.append(f"Manifest {split} records must be a list")
            continue
        seen = set()
        for record in records:
            if not isinstance(record, dict):
                errors.append(f"Invalid manifest record in {split}")
                continue
            filename, group, categories = record.get("filename"), record.get("group"), record.get("class_ids")
            if not isinstance(filename, str) or not filename or "/" in filename or "\\" in filename or Path(filename).suffix.lower() not in IMAGE_SUFFIXES:
                errors.append(f"Invalid manifest filename in {split}: {filename}")
                continue
            if filename in seen:
                errors.append(f"Duplicate manifest filename: {split}/{filename}")
                continue
            seen.add(filename)
            if isinstance(group, bool) or not isinstance(group, int) or group < 0:
                errors.append(f"Invalid manifest group: {split}/{filename}")
                continue
            if not isinstance(categories, list) or any(isinstance(c, bool) or not isinstance(c, int) or not 0 <= c < class_count for c in categories):
                errors.append(f"Invalid manifest class IDs: {split}/{filename}")
                continue
            hashes = ("sha256", "label_sha256") if version == 1 or "label_sha256" in record else ("sha256",)
            if any(not isinstance(record.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", record[key]) for key in hashes):
                errors.append(f"Invalid manifest content hash: {split}/{filename}")
                continue
            cleaned["splits"][split].append(record)
    return cleaned


def validate(config: dict) -> tuple[dict, dict[str, list[tuple[Path, list]]]]:
    from PIL import Image
    report = {"checked_at": utc_now(), "classes": config["names"], "splits": {}, "errors": []}
    seen_hashes, split_samples = {}, {}
    manifest_path = Path(config["path"]) / "manifest.json"
    manifest = read_manifest(manifest_path, len(config["names"]), report["errors"])
    seen_groups = {}
    for split in SPLITS:
        image_dir, label_dir = Path(config[split]), Path(config["path"]) / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            report["errors"].append(f"Missing image/label directory for {split}")
            split_samples[split] = []
            continue
        images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        samples, counts, image_counts, empty = [], Counter(), Counter(), 0
        stems = [image.stem for image in images]
        if len(set(stems)) != len(stems):
            report["errors"].append(f"Duplicate image stems in {split}")
        orphaned = {p.stem for p in label_dir.glob("*.txt")} - set(stems)
        if orphaned:
            report["errors"].append(f"Orphan labels in {split}: {sorted(orphaned)}")
        actual, label_info = {}, {}
        for image in images:
            try:
                with Image.open(image) as opened:
                    opened.verify()
                digest = sha256(image)
                actual[image.name] = digest
                label_path = label_dir / (image.stem + ".txt")
                labels = read_labels(label_path, len(config["names"]))
                if digest in seen_hashes:
                    report["errors"].append(f"Duplicate image bytes: {seen_hashes[digest]} and {split}/{image.name}")
                seen_hashes[digest] = f"{split}/{image.name}"
                class_ids = [int(label[0]) for label in labels]
                label_info[image.name] = {"class_ids": class_ids, "sha256": sha256(label_path)}
                counts.update(class_ids)
                image_counts.update(set(class_ids))
                empty += not labels
                samples.append((image, labels))
            except (ValueError, OSError) as error:
                report["errors"].append(f"{split}/{image.name}: {error}")
        if manifest is not None:
            records = manifest["splits"][split]
            if {record["filename"] for record in records} != {image.name for image in images}:
                report["errors"].append(f"Files disagree with the frozen {split} manifest")
            split_path = Path(config["path"]) / "splits" / f"{split}.txt"
            try:
                if split_path.read_text(encoding="utf-8").splitlines() != [record["filename"] for record in records]:
                    report["errors"].append(f"Split list disagrees with the frozen {split} manifest")
            except (OSError, ValueError) as error:
                report["errors"].append(f"Cannot read {split} split list: {error}")
            for record in records:
                if actual.get(record["filename"]) != record["sha256"]:
                    report["errors"].append(f"Image content changed: {split}/{record['filename']}")
                label = label_info.get(record["filename"])
                if label is not None:
                    if label["class_ids"] != record["class_ids"]:
                        report["errors"].append(f"Label classes changed: {split}/{record['filename']}")
                    if "label_sha256" in record and label["sha256"] != record["label_sha256"]:
                        report["errors"].append(f"Label content changed: {split}/{record['filename']}")
                group = record["group"]
                if group in seen_groups and seen_groups[group] != split:
                    report["errors"].append(f"Group {group} crosses data splits")
                seen_groups[group] = split
        if not images:
            report["errors"].append(f"Empty split: {split}")
        for index, name in config["names"].items():
            if counts[index] == 0:
                report["errors"].append(f"No {name} instances in {split}; class evaluation would be unavailable")
        report["splits"][split] = {"images": len(images), "negative_images": empty, "class_instances": {config["names"][i]: counts[i] for i in config["names"]}, "images_containing_class": {config["names"][i]: image_counts[i] for i in config["names"]}}
        split_samples[split] = samples
    if not report["errors"]:
        try:
            report["dataset_fingerprint"] = dataset_fingerprint(config)
        except OSError as error:
            report["errors"].append(f"Cannot fingerprint dataset: {error}")
    report["ok"] = not report["errors"]
    return report, split_samples


def draw_sheets(samples: dict, names: dict, destination: Path, count: int, seed: int) -> None:
    from PIL import Image, ImageDraw, ImageOps
    if count < 1:
        raise ValueError("Sample count must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    colors = [(30, 180, 60), (230, 60, 50), (240, 170, 20)]
    for split, items in samples.items():
        rng = random.Random(seed)
        shuffled = list(items)
        rng.shuffle(shuffled)
        selected, seen = [], set()
        # Include available classes before adding random samples.
        for category in names:
            for item in shuffled:
                if item[0] not in seen and any(int(label[0]) == category for label in item[1]):
                    selected.append(item)
                    seen.add(item[0])
                    break
        selected.extend(item for item in shuffled if item[0] not in seen)
        selected = selected[:max(count, len(names))]
        if not selected:
            continue
        cell_width, cell_height, columns = 320, 260, 4
        sheet = Image.new("RGB", (columns * cell_width, math.ceil(len(selected) / columns) * cell_height), "white")
        selected_names = []
        for index, (path, labels) in enumerate(selected):
            with Image.open(path) as image:
                image = image.convert("RGB")
                draw = ImageDraw.Draw(image)
                width, height = image.size
                for category, x, y, w, h in labels:
                    bounds = ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)
                    color = colors[int(category) % len(colors)]
                    draw.rectangle(bounds, outline=color, width=max(2, width // 300))
                    draw.text((bounds[0], max(0, bounds[1] - 12)), names[int(category)], fill=color)
                thumb = ImageOps.contain(image, (cell_width - 8, cell_height - 26))
            left, top = (index % columns) * cell_width, (index // columns) * cell_height
            sheet.paste(thumb, (left + 4, top + 4))
            ImageDraw.Draw(sheet).text((left + 4, top + cell_height - 20), path.name, fill="black")
            selected_names.append(path.name)
        sheet.save(destination / f"{split}_labels.jpg", quality=92)
        write_json(destination / f"{split}_sampled_images.json", selected_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check all splits and save visual label sheets")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--output", default="results/data_check")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = dataset_config(project_path(args.data))
    report, samples = validate(config)
    output = project_path(args.output)
    write_json(output / "report.json", report)
    if not report["ok"]:
        raise ValueError(f"Dataset check failed. See {output / 'report.json'}")
    draw_sheets(samples, config["names"], output, args.samples, args.seed)
    print(f"Dataset checks passed. Inspect the sheets in {output}")


if __name__ == "__main__":
    main()
