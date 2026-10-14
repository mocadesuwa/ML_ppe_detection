"""Prepare traceable head crops from explicit pixel boxes; no model is used."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from .common import IMAGE_SUFFIXES, config_path, project_path, publish_path, sha256, utc_now, validate_run_name, write_json


def unique_fields(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Repeated manifest field: {key}")
        result[key] = value
    return result


def read_plan(manifest: Path) -> dict:
    """Read and validate the original crop specification without writing files."""
    from PIL import Image

    manifest = manifest.resolve()
    content = manifest.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    data = json.loads(content.decode("utf-8-sig"), object_pairs_hook=unique_fields)
    if not isinstance(data, dict) or set(data) != {"format_version", "items"}:
        raise ValueError("Manifest must contain format_version and items only")
    if type(data["format_version"]) is not int or data["format_version"] != 1:
        raise ValueError("Unsupported manifest format_version; expected integer 1")
    if not isinstance(data["items"], list) or not data["items"]:
        raise ValueError("Manifest items must be a non-empty list")
    sources, groups, identifiers, items = {}, {}, set(), []
    required = {"id", "source", "group", "box_xyxy"}
    for item in data["items"]:
        if not isinstance(item, dict) or not required <= set(item) <= required | {"label"}:
            raise ValueError("Each crop requires id, source, group and box_xyxy; label is optional")
        validate_run_name(item["id"])
        identifier = item["id"].casefold()
        if identifier in identifiers:
            raise ValueError(f"Repeated crop id: {item['id']}")
        identifiers.add(identifier)
        group = item["group"]
        if not isinstance(group, str) or not group.strip() or group != group.strip():
            raise ValueError("group must be a non-empty string without surrounding whitespace")
        if "label" in item and (not isinstance(item["label"], str) or not item["label"].strip()):
            raise ValueError("Optional label must be a non-empty string")
        source = config_path(item["source"], "source", manifest.parent)
        if source not in sources:
            if source.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported source image: {source}")
            source_digest = sha256(source)
            with Image.open(source) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError(f"Head crops require static images: {source}")
                if image.getexif().get(274, 1) != 1:
                    raise ValueError("Normalize EXIF orientation and update boxes before cropping")
                image.load()
                width, height = image.size
            if sha256(source) != source_digest:
                raise ValueError(f"Source image changed while planning: {source}")
            sources[source] = {"path": str(source), "sha256": source_digest, "image_size": {"width": width, "height": height}}
        info = sources[source]
        box = item["box_xyxy"]
        if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
            raise ValueError("box_xyxy must contain four integer pixel coordinates")
        x1, y1, x2, y2 = box
        size = info["image_size"]
        if not (0 <= x1 < x2 <= size["width"] and 0 <= y1 < y2 <= size["height"]):
            raise ValueError(f"Empty or out-of-bounds head box: {box}")
        source_hash = info["sha256"]
        if source_hash in groups and groups[source_hash] != group:
            raise ValueError("All crops of identical source content must use the same group")
        groups[source_hash] = group
        record = {**item, "source": str(source), "source_sha256": source_hash,
                  "source_size": size, "crop_size": {"width": x2 - x1, "height": y2 - y1},
                  "output_file": f"images/{item['id']}.png"}
        items.append(record)
    plan = {"manifest": str(manifest), "manifest_sha256": digest,
            "sources": list(sources.values()), "items": items}
    require_unchanged_inputs(plan)
    return plan


def build_plan(manifest: Path, output: Path) -> dict:
    """Validate every box and source before creating any output directory."""
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Crop output already exists: {output}")
    plan = read_plan(manifest)
    for source in plan["sources"]:
        directory = Path(source["path"]).parent
        if output == directory or directory in output.parents:
            raise ValueError("Crop output must be outside the source image directory")
    return {**plan, "output": str(output)}


def require_unchanged_inputs(plan: dict) -> None:
    if sha256(Path(plan["manifest"])) != plan["manifest_sha256"]:
        raise ValueError("Crop manifest changed after planning")
    for source in plan["sources"]:
        if sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"Source image changed after planning: {source['path']}")


def prepare(plan: dict) -> dict:
    """Publish a new crop directory only after every crop and content check succeeds."""
    from PIL import Image

    output = Path(plan["output"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Crop output already exists: {output}")
    require_unchanged_inputs(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".head-crops-", dir=output.parent) as temporary:
        staged = Path(temporary) / "prepared"
        (staged / "images").mkdir(parents=True)
        records = []
        for item in plan["items"]:
            destination = staged / item["output_file"]
            with Image.open(item["source"]) as image:
                with image.crop(tuple(item["box_xyxy"])).convert("RGB") as crop:
                    crop.info.clear()
                    crop.save(destination, format="PNG")
            records.append({**item, "output_sha256": sha256(destination)})
        require_unchanged_inputs(plan)
        report = {"format_version": 1, "purpose": "goggles_head_crops", "status": "complete", "created_at": utc_now(),
                  "source_manifest": plan["manifest"], "source_manifest_sha256": plan["manifest_sha256"],
                  "sources": plan["sources"], "crops": records, "visual_review_completed": False}
        write_json(staged / "crops_manifest.json", report)
        publish_path(staged, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON manifest with explicit head boxes and source groups")
    parser.add_argument("--output", required=True, help="New output directory outside the original image directories")
    parser.add_argument("--dry-run", action="store_true", help="Read and check sources and boxes, print the plan, write nothing")
    args = parser.parse_args()
    try:
        plan = build_plan(config_path(args.manifest, "manifest", project_path(".")), config_path(args.output, "output", project_path(".")))
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        report = prepare(plan)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Prepared {len(report['crops'])} head crops: {plan['output']}")


if __name__ == "__main__":
    main()
