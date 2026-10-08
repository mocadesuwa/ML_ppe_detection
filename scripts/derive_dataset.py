"""Apply reviewed decisions to a new VOC copy, preserving every original file."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.review_raw_dataset import inspect, snapshot
from src.common import sha256, utc_now, write_json
from src.prepare_dataset import convert_box


def derive(raw: Path, output: Path, decisions_path: Path) -> dict:
    raw, output = raw.resolve(), output.resolve()
    if output == raw or raw in output.parents or output in raw.parents:
        raise ValueError("Derived output must be separate from the original directory")
    if output.exists():
        raise FileExistsError("Derived output already exists; choose a new version")
    before, fingerprint = snapshot(raw)
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    decision_hash = sha256(decisions_path)
    if decisions.get("format_version") != 1 or decisions.get("source_fingerprint") != fingerprint:
        raise ValueError("Decisions do not match this original dataset snapshot")
    if decisions.get("coordinates") != "voc_one_based_inclusive":
        raise ValueError("This reviewed derivation requires explicit VOC inclusive coordinates")
    report = inspect(raw)
    if report["file_errors"]:
        raise ValueError("Resolve source file errors before derivation")
    records = {r["filename"]: r for r in report["records"]}
    removed, repairs = {}, {}

    def checked_entry(entry: dict) -> dict:
        name = entry.get("filename")
        if name not in records:
            raise ValueError(f"Unknown decision filename: {name}")
        record = records[name]
        if any(entry.get(key) != record[key] for key in ("image_sha256", "annotation_sha256")):
            raise ValueError(f"Decision hashes differ from source: {name}")
        if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            raise ValueError(f"Decision must state a reason: {name}")
        return record

    for entry in decisions.get("excluded_images", []):
        checked_entry(entry)
        name = entry["filename"]
        if name in removed:
            raise ValueError(f"Repeated removal: {name}")
        removed[name] = {**entry, "action": "exclude_whole_image"}
    for entry in decisions.get("duplicate_removals", []):
        record = checked_entry(entry)
        name, keeper = entry["filename"], entry.get("retained")
        if name in removed or keeper not in records or keeper in removed or keeper == name:
            raise ValueError(f"Invalid duplicate decision: {name}")
        retained = records[keeper]
        if record["image_sha256"] != retained["image_sha256"]:
            raise ValueError("Duplicate decision must refer to identical image bytes")
        if Counter(o["class"] for o in record["objects"]) != Counter(o["class"] for o in retained["objects"]):
            raise ValueError("Duplicate class counts differ; explicit semantic resolution is required")
        removed[name] = {**entry, "action": "remove_exact_duplicate"}
    if any(e.get("retained") in removed for e in decisions.get("duplicate_removals", [])):
        raise ValueError("A duplicate keeper must remain in the derived dataset")
    for entry in decisions.get("box_repairs", []):
        record = checked_entry(entry)
        index = entry.get("object_index")
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(record["objects"]):
            raise ValueError("Repair object index must be a valid one-based integer")
        key = entry["filename"], index
        if key in repairs:
            raise ValueError("Repeated box repair")
        original = record["objects"][index - 1]["box"]
        # Only the individually reviewed one-pixel right-edge errors are allowed.
        expected = [*original[:2], record["width"], original[3]]
        if entry.get("before") != original or original[2] != record["width"] + 1 or entry.get("after") != expected:
            raise ValueError("Repair does not match the reviewed one-pixel xmax policy")
        convert_box(tuple(expected), record["width"], record["height"], decisions["coordinates"])
        repairs[key] = {**entry, "applied": entry["filename"] not in removed}

    # Validate every retained box before creating any derived file.
    for name, record in records.items():
        if name in removed:
            continue
        for obj in record["objects"]:
            box = repairs.get((name, obj["index"]), {}).get("after", obj["box"])
            convert_box(tuple(box), record["width"], record["height"], decisions["coordinates"])

    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "annotations").mkdir()
    selected, counts = [], Counter()
    for name, record in records.items():
        if name in removed:
            continue
        image = output / "images" / name
        annotation_name = Path(record["annotation"]).name
        annotation = output / "annotations" / annotation_name
        shutil.copy2(raw / "images" / name, image)
        source_xml = raw / "annotations" / annotation_name
        changes = [e for (filename, _), e in repairs.items() if filename == name]
        if changes:
            tree = ET.parse(source_xml)
            for entry in changes:
                obj = tree.getroot().findall("object")[entry["object_index"] - 1]
                obj.find("bndbox/xmax").text = str(entry["after"][2])
            tree.write(annotation, encoding="utf-8", xml_declaration=True)
        else:
            shutil.copy2(source_xml, annotation)
        if sha256(image) != record["image_sha256"]:
            raise RuntimeError("Copied image differs from source")
        counts.update(obj["class"] for obj in record["objects"])
        selected.append({"filename": name, "image_sha256": sha256(image), "annotation": annotation_name,
                         "original_annotation_sha256": record["annotation_sha256"], "derived_annotation_sha256": sha256(annotation)})
    if snapshot(raw) != (before, fingerprint) or sha256(decisions_path) != decision_hash:
        raise RuntimeError("Original files or decisions changed during derivation; output is unconfirmed")
    ledger = {"format_version": 1, "completed_at": utc_now(), "status": "complete", "source_unchanged": True,
              "source_fingerprint": fingerprint, "source_files": before, "decisions_sha256": decision_hash,
              "coordinates": decisions["coordinates"], "retained_images": len(selected), "class_objects": dict(counts),
              "removed_images": list(removed.values()), "box_repairs": list(repairs.values()), "selected_files": selected}
    write_json(output / "derivation.json", ledger)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="dataset/raw/face-mask-detection")
    parser.add_argument("--output", default="dataset/derived/source_v1")
    parser.add_argument("--decisions", default="configs/source_decisions_v1.json")
    args = parser.parse_args()
    ledger = derive(Path(args.raw), Path(args.output), Path(args.decisions))
    print(json.dumps({k: ledger[k] for k in ("status", "source_unchanged", "retained_images", "class_objects")}, indent=2))


if __name__ == "__main__":
    main()
