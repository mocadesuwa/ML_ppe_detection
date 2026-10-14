"""Plan reproducible, grouped head-crop splits without choosing a model format."""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from .check_head_crops import inspect_crops
from .common import SPLITS, config_path, project_path, publish_path, utc_now, write_json
from .prepare_dataset import split_groups


def crop_groups(crops: list[dict]) -> list[list[int]]:
    """Join explicit groups, identical sources and identical decoded crop pixels."""
    parents = list(range(len(crops)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    representatives = {}
    labels = {}
    for index, item in enumerate(crops):
        pixels = item["pixel_sha256"]
        if pixels in labels and labels[pixels] != item["label"]:
            raise ValueError("Identical crop pixels have conflicting labels; resolve before splitting")
        labels[pixels] = item["label"]
        for field in ("group", "source_sha256", "pixel_sha256"):
            key = (field, item[field])
            if key in representatives:
                parents[find(index)] = find(representatives[key])
            else:
                representatives[key] = index
    groups = {}
    for index in range(len(crops)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def build_split(root: Path, classes: list[str], ratios: list[float], seed: int) -> dict:
    if (not isinstance(classes, list) or not classes
            or any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError("Explicit class names must be non-empty, unique strings without surrounding whitespace")
    checked = inspect_crops(root)
    crops = checked["crops"]
    if checked["unlabeled_count"] or set(checked["label_counts"]) != set(classes):
        raise ValueError("Every crop must have an explicit label; labels must match all supplied class names exactly")
    groups = crop_groups(crops)
    samples = [{"labels": [[classes.index(item["label"])]]} for item in crops]
    assignments = split_groups(groups, samples, ratios, seed, len(classes))
    group_ids = {index: min(crops[i]["id"] for i in group) for group in groups for index in group}
    splits = {split: [{"id": crops[index]["id"], "output_file": crops[index]["output_file"],
                       "output_sha256": crops[index]["output_sha256"], "label": crops[index]["label"],
                       "source_group": crops[index]["group"], "split_group": group_ids[index]}
                      for index in assignments[split]] for split in SPLITS}
    return {"format_version": 1, "purpose": "goggles_head_split_plan", "status": "planned",
            "created_at": utc_now(), "crop_directory": checked["crop_directory"],
            "source_directories": sorted({str(Path(source["path"]).parent) for source in checked["sources"]}),
            "crop_fingerprint": checked["crop_fingerprint"], "classes": classes, "seed": seed,
            "requested_ratios": dict(zip(SPLITS, ratios)), "independent_groups": len(groups),
            "splits": splits, "counts": {split: {"crops": len(items), "labels": dict(Counter(item["label"] for item in items))}
                                         for split, items in splits.items()},
            "grouping_limitations": "Only explicit groups, identical source bytes and identical crop pixels are joined. Review person/sequence/near-duplicate relations manually.",
            "visual_review_completed": False, "training_ready": False}


def validate_output(plan: dict, output: Path) -> Path:
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Split output exists; choose a new directory")
    protected = [Path(plan["crop_directory"])] + [Path(directory) for directory in plan["source_directories"]]
    if any(output == directory or directory in output.parents for directory in protected):
        raise ValueError("Split output must be outside crop and source directories")
    return output


def publish_split(plan: dict, output: Path) -> None:
    """Revalidate the plan before publishing a new directory; never copy images."""
    expected = build_split(Path(plan["crop_directory"]), plan["classes"], [plan["requested_ratios"][split] for split in SPLITS], plan["seed"])
    if {key: value for key, value in expected.items() if key != "created_at"} != {key: value for key, value in plan.items() if key != "created_at"}:
        raise ValueError("Split plan or crop inputs changed before publication")
    output = validate_output(expected, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".head-crops-", dir=output.parent) as temporary:
        staged = Path(temporary) / "prepared"
        write_json(staged / "splits_manifest.json", plan)
        if inspect_crops(Path(plan["crop_directory"]))["crop_fingerprint"] != plan["crop_fingerprint"]:
            raise ValueError("Crop inputs changed during split publication")
        publish_path(staged, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", required=True, help="Directory created by src.prepare_head_crops")
    parser.add_argument("--classes", nargs="+", required=True, help="Explicit crop labels; no business labels are inferred")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="New split-plan directory outside all input directories")
    parser.add_argument("--dry-run", action="store_true", help="Read inputs and print assignments without writing")
    args = parser.parse_args()
    try:
        plan = build_split(config_path(args.crops, "crops", project_path(".")), args.classes, args.ratios, args.seed)
        output = validate_output(plan, config_path(args.output, "output", project_path(".")))
        if not args.dry_run:
            publish_split(plan, output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
