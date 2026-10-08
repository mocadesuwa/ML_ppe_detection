"""Check labels, cross-split duplicates and generate inspection sheets."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from .common import IMAGE_SUFFIXES, SPLITS, dataset_config, dataset_fingerprint, project_path, sha256, utc_now, write_json


def read_labels(path: Path, class_count: int) -> list[list[float]]:
    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Line {line_number}: expected five fields")
        values = list(map(float, parts))
        category, x, y, width, height = values
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Line {line_number}: non-finite value")
        if category != int(category) or not 0 <= category < class_count:
            raise ValueError(f"Line {line_number}: invalid class ID")
        tolerance = 1e-7
        if not (width > 0 and height > 0 and x - width / 2 >= -tolerance and x + width / 2 <= 1 + tolerance and y - height / 2 >= -tolerance and y + height / 2 <= 1 + tolerance):
            raise ValueError(f"Line {line_number}: invalid normalized box")
        labels.append(values)
    return labels


def validate(config: dict) -> tuple[dict, dict[str, list[tuple[Path, list]]]]:
    from PIL import Image
    report = {"checked_at": utc_now(), "classes": config["names"], "splits": {}, "errors": []}
    seen_hashes, split_samples = {}, {}
    manifest_path = Path(config["path"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
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
        for image in images:
            try:
                with Image.open(image) as opened:
                    opened.verify()
                labels = read_labels(label_dir / (image.stem + ".txt"), len(config["names"]))
                digest = sha256(image)
                if digest in seen_hashes:
                    report["errors"].append(f"Duplicate image bytes: {seen_hashes[digest]} and {split}/{image.name}")
                seen_hashes[digest] = f"{split}/{image.name}"
                class_ids = [int(label[0]) for label in labels]
                counts.update(class_ids)
                image_counts.update(set(class_ids))
                empty += not labels
                samples.append((image, labels))
            except (ValueError, OSError) as error:
                report["errors"].append(f"{split}/{image.name}: {error}")
        if manifest:
            records = manifest["splits"][split]
            if {record["filename"] for record in records} != {image.name for image in images}:
                report["errors"].append(f"Files disagree with the frozen {split} manifest")
            actual = {image.name: sha256(image) for image in images}
            for record in records:
                if actual.get(record["filename"]) != record["sha256"]:
                    report["errors"].append(f"Image content changed: {split}/{record['filename']}")
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
        report["dataset_fingerprint"] = dataset_fingerprint(config)
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
