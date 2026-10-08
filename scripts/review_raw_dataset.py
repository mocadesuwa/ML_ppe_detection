"""Read original VOC data and produce audit evidence; never repair source files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import IMAGE_SUFFIXES, sha256, utc_now, write_json
from src.prepare_dataset import convert_box, difference_hash

CLASSES = ("with_mask", "without_mask", "mask_weared_incorrect")
COLORS = {"with_mask": "#12d698", "without_mask": "#ff6868", "mask_weared_incorrect": "#ffcc33"}


def snapshot(raw: Path) -> tuple[list[dict], str]:
    files = [{"path": path.relative_to(raw).as_posix(), "sha256": sha256(path)} for path in sorted(raw.rglob("*")) if path.is_file()]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return files, digest


def inspect(raw: Path) -> dict:
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
                if name not in CLASSES:
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
    return {"image_count": len(images), "annotation_count": len(annotations), "object_count": sum(class_counts.values()), "class_objects": dict(class_counts), "images_with_class": dict(images_with_class), "geometry": dict(geometry), "invalid_boxes": invalid, "file_errors": errors, "exact_duplicate_groups": exact, "records": records}


def make_sheets(raw: Path, output: Path, report: dict) -> dict:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    lookup = {r["filename"]: r for r in report["records"]}
    chosen = list(dict.fromkeys(item["filename"] for item in report["invalid_boxes"]))
    minority = [r for r in report["records"] if any(o["class"] == CLASSES[2] for o in r["objects"])]
    candidates = minority[:10] + sorted(report["records"], key=lambda r: len(r["objects"]), reverse=True)[:4] + report["records"]
    for record in candidates:
        if len(chosen) >= 30:
            break
        if record["filename"] not in chosen:
            chosen.append(record["filename"])
    pages = []
    for page, start in enumerate(range(0, len(chosen), 6), 1):
        sheet = Image.new("RGB", (1000, 1350), "#17232b")
        draw = ImageDraw.Draw(sheet)
        for slot, filename in enumerate(chosen[start:start + 6]):
            record = lookup[filename]
            left, top = (slot % 2) * 500, (slot // 2) * 450
            with Image.open(raw / "images" / filename) as opened:
                annotated = opened.convert("RGB")
            overlay = ImageDraw.Draw(annotated)
            for obj in record["objects"]:
                box = obj["box"]
                overlay.rectangle(box, outline=COLORS.get(obj["class"], "white"), width=2)
                overlay.text((box[0], max(0, box[1] - 11)), str(obj["index"]), fill="white")
            annotated.thumbnail((480, 365))
            sheet.paste(annotated, (left + 10, top + 48))
            draw.text((left + 10, top + 6), f"{start + slot + 1}. {filename} ({len(record['objects'])} boxes)", font=font, fill="white")
            draw.text((left + 10, top + 28), "green=with | red=without | yellow=incorrect; original boxes", font=small, fill="#dddddd")
        name = f"overview_{page:02d}.png"
        sheet.save(output / name)
        pages.append(name)

    def crops(items: list[dict], prefix: str) -> list[str]:
        crop_pages = []
        for page, start in enumerate(range(0, len(items), 12), 1):
            sheet = Image.new("RGB", (960, 1000), "#17232b")
            draw = ImageDraw.Draw(sheet)
            for slot, item in enumerate(items[start:start + 12]):
                left, top = slot % 3 * 320, slot // 3 * 250
                filename, box = item["filename"], item["box"]
                record = lookup[filename]
                with Image.open(raw / "images" / filename) as opened:
                    annotated = opened.convert("RGB")
                overlay = ImageDraw.Draw(annotated)
                overlay.rectangle(box, outline=COLORS[item["class"]], width=1)
                margin = max(12, round(max(box[2] - box[0], box[3] - box[1]) * .6))
                region = (max(0, int(box[0]) - margin), max(0, int(box[1]) - margin), min(record["width"], int(box[2]) + margin), min(record["height"], int(box[3]) + margin))
                cropped = annotated.crop(region)
                scale = min(300 / cropped.width, 190 / cropped.height)
                cropped = cropped.resize((max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))), Image.Resampling.NEAREST)
                sheet.paste(cropped, (left + 10, top + 52))
                draw.text((left + 10, top + 4), f"{filename} #{item['index']}", font=small, fill="white")
                draw.text((left + 10, top + 22), item["class"], font=small, fill=COLORS[item["class"]])
                draw.text((left + 10, top + 38), str([int(v) for v in box]), font=small, fill="#dddddd")
            name = f"{prefix}_{page:02d}.png"
            sheet.save(output / name)
            crop_pages.append(name)
        return crop_pages

    minority_items = [{"filename": r["filename"], **obj} for r in report["records"] for obj in r["objects"] if obj["class"] == CLASSES[2]]
    duplicate_pages = []
    for page, group in enumerate(report["exact_duplicate_groups"], 1):
        sheet = Image.new("RGB", (1000, 550), "#17232b")
        draw = ImageDraw.Draw(sheet)
        for slot, filename in enumerate(group["filenames"]):
            record = lookup[filename]
            with Image.open(raw / "images" / filename) as opened:
                annotated = opened.convert("RGB")
            overlay = ImageDraw.Draw(annotated)
            for obj in record["objects"]:
                overlay.rectangle(obj["box"], outline=COLORS[obj["class"]], width=1)
                overlay.text((obj["box"][0], max(0, obj["box"][1] - 10)), str(obj["index"]), fill="white")
            annotated.thumbnail((480, 480))
            sheet.paste(annotated, (slot * 500 + 10, 55))
            draw.text((slot * 500 + 10, 8), filename, font=font, fill="white")
            draw.text((slot * 500 + 10, 30), f"{len(record['objects'])} boxes; original labels", font=font, fill="#dddddd")
        name = f"duplicate_{page:02d}.png"
        sheet.save(output / name)
        duplicate_pages.append(name)
    return {"selected_images": chosen, "overview_sheets": pages, "invalid_box_sheets": crops(report["invalid_boxes"], "invalid"), "incorrect_class_sheets": crops(minority_items, "incorrect"), "duplicate_sheets": duplicate_pages, "note": "Crops are clipped and enlarged for display only; no XML boxes or raw pixels are changed. Sheets do not imply completed visual review."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="dataset/raw/face-mask-detection")
    parser.add_argument("--output", required=True, help="New directory outside original data")
    args = parser.parse_args()
    raw, output = Path(args.raw).resolve(), Path(args.output).resolve()
    if output == raw or raw in output.parents:
        parser.error("Output must be outside the original data directory")
    if output.exists():
        parser.error("Output already exists; choose a new audit directory")
    before, fingerprint = snapshot(raw)
    report = inspect(raw)
    output.mkdir(parents=True)
    report.update(checked_at=utc_now(), raw_directory=str(raw), source_files=before, source_fingerprint=fingerprint, visual_review_completed=False)
    report["sheets"] = make_sheets(raw, output, report)
    after, after_fingerprint = snapshot(raw)
    if before != after or fingerprint != after_fingerprint:
        raise RuntimeError("Original files changed during audit; report is not confirmed")
    report["source_unchanged"] = True
    write_json(output / "audit.json", report)
    print(json.dumps({k: report[k] for k in ("image_count", "annotation_count", "object_count", "class_objects", "images_with_class", "geometry", "exact_duplicate_groups", "source_fingerprint", "source_unchanged")}, indent=2))
    print(f"Invalid objects: {len(report['invalid_boxes'])}; file errors: {len(report['file_errors'])}; evidence: {output}")


if __name__ == "__main__":
    main()
