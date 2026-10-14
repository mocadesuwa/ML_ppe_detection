"""Read-only verification of head crops against their original images and boxes."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .common import config_path, project_path, sha256
from .prepare_head_crops import read_plan, require_unchanged_inputs, unique_fields


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def require_inventory(root: Path, files: set[str]) -> None:
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Crop directory must not contain symbolic links")
    if {path.relative_to(root).as_posix() for path in paths} != files | {"images"}:
        raise ValueError("Crop directory has missing or unexpected files/directories")


def inspect_crops(root: Path) -> dict:
    """Check exact provenance and pixels; this does not perform semantic review."""
    from PIL import Image

    root = root.resolve()
    ledger = root / "crops_manifest.json"
    content = ledger.read_bytes()
    ledger_hash = hashlib.sha256(content).hexdigest()
    data = json.loads(content.decode("utf-8-sig"), object_pairs_hook=unique_fields)
    fields = {"format_version", "purpose", "status", "created_at", "source_manifest",
              "source_manifest_sha256", "sources", "crops", "visual_review_completed"}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError("Invalid crops_manifest schema")
    if (type(data["format_version"]) is not int or data["format_version"] != 1
            or data["purpose"] != "goggles_head_crops" or data["status"] != "complete"
            or data["visual_review_completed"] is not False
            or not isinstance(data["created_at"], str) or not data["created_at"].strip()):
        raise ValueError("Unsupported or incomplete crop manifest")
    manifest = config_path(data["source_manifest"], "source_manifest", root)
    plan = read_plan(manifest)
    if (data["source_manifest"] != plan["manifest"]
            or data["source_manifest_sha256"] != plan["manifest_sha256"]
            or canonical(data["sources"]) != canonical(plan["sources"])):
        raise ValueError("Crop provenance no longer matches the source manifest/images")
    if not isinstance(data["crops"], list) or len(data["crops"]) != len(plan["items"]):
        raise ValueError("Crop records do not match the source manifest")
    files = {"crops_manifest.json"} | {item["output_file"] for item in plan["items"]}
    require_inventory(root, files)
    records = []
    for item, expected in zip(data["crops"], plan["items"]):
        if (not isinstance(item, dict) or set(item) != set(expected) | {"output_sha256"}
                or canonical({key: value for key, value in item.items() if key != "output_sha256"}) != canonical(expected)):
            raise ValueError("Crop record differs from its original specification")
        path = root / expected["output_file"]
        if sha256(path) != item["output_sha256"]:
            raise ValueError(f"Crop hash mismatch: {item['id']}")
        with Image.open(path) as crop, Image.open(item["source"]) as original:
            size = item["crop_size"]
            if (crop.format != "PNG" or crop.mode != "RGB" or getattr(crop, "n_frames", 1) != 1
                    or crop.getexif().get(274, 1) != 1 or crop.size != (size["width"], size["height"])):
                raise ValueError(f"Invalid crop format/size: {item['id']}")
            pixels = crop.tobytes()
            if pixels != original.crop(tuple(item["box_xyxy"])).convert("RGB").tobytes():
                raise ValueError(f"Crop pixels differ from the original box: {item['id']}")
            pixel_hash = hashlib.sha256(canonical(size).encode("utf-8") + b"\0" + pixels).hexdigest()
        records.append({**item, "pixel_sha256": pixel_hash})
    require_unchanged_inputs(plan)
    require_inventory(root, files)
    if sha256(ledger) != ledger_hash or any(sha256(root / item["output_file"]) != item["output_sha256"] for item in records):
        raise ValueError("Crop content changed during verification")
    fingerprint = hashlib.sha256(canonical({"manifest_sha256": ledger_hash, "crops": records}).encode("utf-8")).hexdigest()
    return {"format_version": 1, "status": "integrity_passed", "crop_directory": str(root),
            "crop_fingerprint": fingerprint, "manifest_sha256": ledger_hash,
            "source_manifest": str(manifest), "sources": plan["sources"], "crop_count": len(records),
            "label_counts": dict(sorted(Counter(item["label"] for item in records if "label" in item).items())),
            "unlabeled_count": sum("label" not in item for item in records), "crops": records,
            "visual_review_completed": False, "training_ready": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", required=True, help="Directory created by src.prepare_head_crops; writes nothing")
    args = parser.parse_args()
    try:
        report = inspect_crops(config_path(args.crops, "crops", project_path(".")))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
