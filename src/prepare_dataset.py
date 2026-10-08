"""Convert VOC annotations, group related images and freeze a YOLO split."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from .common import IMAGE_SUFFIXES, SPLITS, load_yaml, project_path, sha256, utc_now, write_json


def convert_box(box: tuple[float, float, float, float], width: int, height: int, convention: str) -> tuple[float, float, float, float]:
    """Normalize a box using positive integer image dimensions.

    VOC endpoints include their pixels, so a single-pixel box is valid.
    Continuous endpoints must enclose a positive area.
    """
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in (width, height)):
        raise ValueError("Image width and height must be positive integers")
    x1, y1, x2, y2 = box
    if convention == "voc_one_based_inclusive":
        x1, y1 = x1 - 1, y1 - 1
    elif convention != "zero_based_continuous":
        raise ValueError(f"Unknown coordinate convention: {convention}")
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("Non-finite bounding box")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid or out-of-bounds box {box} in {width}x{height} image")
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)


def difference_hash(image) -> int:
    from PIL import Image
    pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def read_samples(raw: Path, classes: dict[str, int], convention: str) -> tuple[list[dict], list[dict]]:
    from PIL import Image
    images: dict[str, Path] = {}
    for image in sorted(p for p in raw.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        if image.stem in images:
            raise ValueError(f"Ambiguous image stem: {image.stem}")
        images[image.stem] = image
    annotations = sorted(raw.rglob("*.xml"))
    if not annotations or not images:
        raise ValueError(f"Expected original images and XML annotations under {raw}")
    samples, errors, seen = [], [], set()
    for annotation in annotations:
        try:
            root = ET.parse(annotation).getroot()
            filename = Path((root.findtext("filename") or annotation.stem).replace("\\", "/")).stem
            image = images.get(filename, images.get(annotation.stem))
            if image is None:
                raise ValueError("No matching image")
            if image.stem in seen:
                raise ValueError("More than one annotation for the same image")
            seen.add(image.stem)
            with Image.open(image) as opened:
                opened.load()
                width, height = opened.size
                dhash = difference_hash(opened)
            xml_width = int(root.findtext("size/width", "0"))
            xml_height = int(root.findtext("size/height", "0"))
            if (xml_width, xml_height) != (width, height):
                raise ValueError(f"XML dimensions {(xml_width, xml_height)} differ from actual {(width, height)}")
            labels = []
            for obj in root.findall("object"):
                name = (obj.findtext("name") or "").strip()
                if name not in classes:
                    raise ValueError(f"Unknown class: {name}")
                box = tuple(float(obj.findtext(f"bndbox/{key}", "nan")) for key in ("xmin", "ymin", "xmax", "ymax"))
                labels.append([classes[name], *convert_box(box, width, height, convention)])
            samples.append({"image": image, "annotation": annotation, "labels": labels, "sha256": sha256(image), "dhash": dhash})
        except (ValueError, ET.ParseError, OSError) as error:
            errors.append({"annotation": str(annotation), "error": str(error)})
    for stem, image in images.items():
        if stem not in seen:
            errors.append({"image": str(image), "error": "Missing annotation; not assumed to be a negative image"})
    return samples, errors


def group_samples(samples: list[dict], distance: int, csv_path: Path | None = None) -> list[list[int]]:
    if not 0 <= distance <= 64:
        raise ValueError("Near-duplicate distance must be between 0 and 64")
    parents = list(range(len(samples)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a: int, b: int) -> None:
        parents[find(a)] = find(b)

    for i, left in enumerate(samples):
        for j in range(i):
            right = samples[j]
            if left["sha256"] == right["sha256"] or (left["dhash"] ^ right["dhash"]).bit_count() <= distance:
                union(i, j)
    if csv_path:
        source_groups: dict[str, str] = {}
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"filename", "group"}.issubset(reader.fieldnames or []):
                raise ValueError("groups CSV must contain filename and group columns")
            for row in reader:
                if row["filename"] in source_groups:
                    raise ValueError(f"Repeated CSV filename: {row['filename']}")
                source_groups[row["filename"]] = row["group"]
        known = {sample["image"].name for sample in samples}
        if set(source_groups) - known:
            raise ValueError("groups CSV contains filenames absent from the usable data")
        representatives: dict[str, int] = {}
        for i, sample in enumerate(samples):
            group = source_groups.get(sample["image"].name)
            if group:
                if group in representatives:
                    union(i, representatives[group])
                else:
                    representatives[group] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(samples)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def split_groups(groups: list[list[int]], samples: list[dict], ratios: list[float], seed: int, class_count: int) -> dict[str, list[int]]:
    import math
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("Three positive split ratios summing to 1 are required")
    if len(groups) < 3:
        raise ValueError("At least three independent groups are needed")
    group_counts = [Counter(int(label[0]) for i in group for label in samples[i]["labels"]) for group in groups]
    totals = sum(group_counts, Counter())
    if any(totals[c] == 0 for c in range(class_count)):
        raise ValueError("The source data does not contain all configured classes")
    total_images = len(samples)
    best = None
    # Multiple deterministic greedy trials balance group sizes and class instances.
    for trial in range(64):
        rng = random.Random(seed + trial)
        order = list(range(len(groups)))
        rng.shuffle(order)
        order.sort(key=lambda i: len(groups[i]), reverse=True)
        assignments = {split: [] for split in SPLITS}
        counts = {split: Counter() for split in SPLITS}
        sizes = {split: 0 for split in SPLITS}
        for index in order:
            candidates = list(SPLITS)
            rng.shuffle(candidates)

            def cost(candidate: str) -> float:
                value = 0.0
                for split, ratio in zip(SPLITS, ratios):
                    size = sizes[split] + (len(groups[index]) if split == candidate else 0)
                    value += ((size - total_images * ratio) / max(total_images * ratio, 1)) ** 2
                    for c in range(class_count):
                        count = counts[split][c] + (group_counts[index][c] if split == candidate else 0)
                        value += ((count - totals[c] * ratio) / max(totals[c] * ratio, 1)) ** 2
                return value

            chosen = min(candidates, key=cost)
            assignments[chosen].extend(groups[index])
            sizes[chosen] += len(groups[index])
            counts[chosen].update(group_counts[index])
        missing = sum(counts[split][c] == 0 for split in SPLITS for c in range(class_count))
        score = missing * 10000 + sum(((sizes[s] - total_images * r) / max(total_images * r, 1)) ** 2 for s, r in zip(SPLITS, ratios))
        score += sum(((counts[s][c] - totals[c] * r) / max(totals[c] * r, 1)) ** 2 for s, r in zip(SPLITS, ratios) for c in range(class_count))
        if best is None or score < best[0]:
            best = score, assignments, missing
    if best[2]:
        raise ValueError("Cannot place every class in all three splits without breaking groups. Check grouping or add minority-class data.")
    return {split: sorted(indices) for split, indices in best[1].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VOC labels and create an immutable grouped YOLO split")
    parser.add_argument("--config", default="configs/prepare.yaml")
    parser.add_argument("--exclude-file", help="Text file listing original image stems to exclude after inspection")
    args = parser.parse_args()
    config = load_yaml(project_path(args.config))
    raw, output = project_path(config["raw"]), project_path(config["output"])
    if (output / "manifest.json").exists() or any((output / "labels").glob("**/*.txt")) or any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in (output / "images").glob("**/*")):
        raise FileExistsError("Prepared data already exists. Use a new output directory/config for a new version.")
    classes = {str(name): int(index) for name, index in config["classes"].items()}
    if sorted(classes.values()) != list(range(len(classes))):
        raise ValueError("Class IDs must start at zero and be contiguous")
    samples, errors = read_samples(raw, classes, config["coordinates"])
    excluded = set()
    if args.exclude_file:
        excluded = {line.strip() for line in project_path(args.exclude_file).read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")}
        known = {sample["image"].stem for sample in samples}
        if excluded - known:
            raise ValueError("Exclude list contains unknown or invalid samples; fix errors before excluding")
        samples = [sample for sample in samples if sample["image"].stem not in excluded]
    write_json(project_path("results/data_preparation/source_check.json"), {"checked_at": utc_now(), "valid_images": len(samples), "errors": errors, "excluded_stems": sorted(excluded)})
    if errors:
        raise ValueError("Source data has errors. See results/data_preparation/source_check.json; no partial split was generated.")
    unique, hashes, duplicates = [], {}, []
    for sample in samples:
        digest = sample["sha256"]
        if digest in hashes:
            previous = hashes[digest]
            if sorted(map(tuple, previous["labels"])) != sorted(map(tuple, sample["labels"])):
                raise ValueError("Identical image bytes have conflicting labels; resolve before splitting")
            duplicates.append({"removed": sample["image"].name, "retained": previous["image"].name})
        else:
            hashes[digest] = sample
            unique.append(sample)
    samples = unique
    groups_csv = project_path(config["groups_csv"]) if config.get("groups_csv") else None
    groups = group_samples(samples, int(config["near_duplicate_distance"]), groups_csv)
    assignments = split_groups(groups, samples, config["ratios"], int(config["seed"]), len(classes))
    group_ids = {index: group_id for group_id, group in enumerate(groups) for index in group}
    manifest = {"created_at": utc_now(), "config": config, "raw_directory": str(raw), "exact_duplicates_removed": duplicates, "excluded_stems": sorted(excluded), "independent_groups": len(groups), "grouping_limitations": "dHash detects similar appearance, not all shared video/person sources. Supply groups_csv where available.", "splits": {}}
    for split in SPLITS:
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for index in assignments[split]:
            sample = samples[index]
            image = sample["image"]
            shutil.copy2(image, image_dir / image.name)
            lines = [f"{int(label[0])} " + " ".join(f"{v:.8f}" for v in label[1:]) for label in sample["labels"]]
            (label_dir / (image.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            records.append({"filename": image.name, "sha256": sample["sha256"], "dhash": str(sample["dhash"]), "group": group_ids[index], "class_ids": [int(label[0]) for label in sample["labels"]]})
        manifest["splits"][split] = records
        split_path = output / "splits" / f"{split}.txt"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text("\n".join(record["filename"] for record in records) + "\n", encoding="utf-8")
    write_json(output / "manifest.json", manifest)
    print(json.dumps({split: len(records) for split, records in manifest["splits"].items()}))
    print("Next: python -m src.check_dataset, inspect sheets, then record the review")


if __name__ == "__main__":
    main()
