"""Audit VOC image/annotation pairs with explicitly supplied class names."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from .common import IMAGE_SUFFIXES, config_path, project_path, sha256, utc_now, write_json
from .prepare_dataset import convert_box, difference_hash


def snapshot(raw: Path) -> tuple[list[dict], str]:
    files = [{"path": path.relative_to(raw).as_posix(), "sha256": sha256(path)} for path in sorted(raw.rglob("*")) if path.is_file()]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return files, digest


def inspect(raw: Path, classes: list[str]) -> dict:
    if not isinstance(classes, list) or not classes:
        raise ValueError("Provide a non-empty list of class names")
    if any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in classes):
        raise ValueError("Class names must be non-empty strings without surrounding whitespace")
    if len(set(classes)) != len(classes):
        raise ValueError("Class names must be distinct")
    from PIL import Image

    images = {p.name: p for p in (raw / "images").iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    annotations = sorted((raw / "annotations").glob("*.xml"))
    records, errors, invalid = [], [], []
    class_counts, images_with_class, geometry = Counter(), Counter(), Counter()
    referenced = set()
    for annotation in annotations:
        try:
            tree = ET.parse(annotation).getroot()
            filename = (tree.findtext("filename") or "").strip()
            if filename not in images:
                raise ValueError(f"Missing declared image: {filename}")
            if filename in referenced:
                raise ValueError(f"Repeated annotation for image: {filename}")
            referenced.add(filename)
            image = images[filename]
            with Image.open(image) as opened:
                opened.load()
                width, height = opened.size
                dhash = difference_hash(opened)
            declared = (int(tree.findtext("size/width", "0")), int(tree.findtext("size/height", "0")))
            if declared != (width, height):
                raise ValueError(f"XML size {declared} differs from image {(width, height)}")
            objects = []
            for index, obj in enumerate(tree.findall("object"), 1):
                name = (obj.findtext("name") or "").strip()
                values = tuple(float(obj.findtext("bndbox/" + key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax"))
                item = {"index": index, "class": name, "box": list(values)}
                if name not in classes:
                    errors.append({"annotation": annotation.name, "object_index": index, "error": f"Unknown class: {name}"})
                for convention in ("zero_based_continuous", "voc_one_based_inclusive"):
                    try:
                        convert_box(values, width, height, convention)
                        item[convention] = True
                    except ValueError:
                        item[convention] = False
                    geometry[convention + "_valid_objects"] += item[convention]
                if not item["zero_based_continuous"] or not item["voc_one_based_inclusive"]:
                    invalid.append({"annotation": annotation.name, "filename": filename, "width": width, "height": height, **item})
                x1, y1, x2, y2 = values
                geometry["xmin_or_ymin_zero"] += x1 == 0 or y1 == 0
                geometry["touches_right_or_bottom"] += x2 == width or y2 == height
                geometry["width_or_height_under_16"] += all(math.isfinite(v) for v in values) and min(x2 - x1, y2 - y1) < 16
                class_counts[name] += 1
                objects.append(item)
            images_with_class.update({obj["class"] for obj in objects})
            records.append({"filename": filename, "annotation": annotation.name, "width": width, "height": height, "image_sha256": sha256(image), "annotation_sha256": sha256(annotation), "dhash": str(dhash), "objects": objects})
        except (ET.ParseError, ValueError, OSError) as error:
            errors.append({"annotation": annotation.name, "error": str(error)})
    for filename in sorted(set(images) - referenced):
        errors.append({"filename": filename, "error": "Missing annotation"})
    duplicates = defaultdict(list)
    for record in records:
        duplicates[record["image_sha256"]].append(record)
    exact = []
    for group in duplicates.values():
        if len(group) > 1:
            labels = [sorted((obj["class"], *obj["box"]) for obj in record["objects"]) for record in group]
            exact.append({"filenames": [r["filename"] for r in group], "conflicting_labels": any(label != labels[0] for label in labels[1:])})
    return {"classes": list(classes), "image_count": len(images), "annotation_count": len(annotations), "object_count": sum(class_counts.values()), "class_objects": dict(class_counts), "images_with_class": dict(images_with_class), "geometry": dict(geometry), "invalid_boxes": invalid, "file_errors": errors, "exact_duplicate_groups": exact, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Directory containing images/ and annotations/")
    parser.add_argument("--classes", nargs="+", required=True, help="Exact class names allowed in the source XML")
    parser.add_argument("--output", required=True, help="New audit directory outside the original files")
    args = parser.parse_args()
    try:
        raw = config_path(args.raw, "raw", project_path("."))
        output = config_path(args.output, "output", project_path("."))
        if output == raw or raw in output.parents:
            raise ValueError("Output must be outside the original data directory")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Output already exists; choose a new audit directory")
        before, fingerprint = snapshot(raw)
        report = inspect(raw, args.classes)
        if snapshot(raw) != (before, fingerprint):
            raise RuntimeError("Original files changed during audit; report is unconfirmed")
        report.update(checked_at=utc_now(), raw_directory=str(raw), source_files=before,
                      source_fingerprint=fingerprint, source_unchanged=True, visual_review_completed=False)
        write_json(output / "audit.json", report)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Audited {report['image_count']} images; file errors: {len(report['file_errors'])}; evidence: {output}")


if __name__ == "__main__":
    main()
