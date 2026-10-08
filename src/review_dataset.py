"""Record the actual annotation/source review, tied to a dataset fingerprint."""
from __future__ import annotations

import argparse
from pathlib import Path

from .check_dataset import validate
from .common import dataset_config, project_path, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a completed visual review; this command does not inspect images for you")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True, help="Actual sampled-image findings and coordinate/class rules")
    parser.add_argument("--license-source", required=True, help="Publisher URL or local license file actually checked")
    args = parser.parse_args()
    config = dataset_config(project_path(args.data))
    report, _ = validate(config)
    if not report["ok"]:
        raise ValueError("Resolve dataset check errors before recording the review")
    write_json(Path(config["path"]) / "quality_review.json", {"reviewed_at": utc_now(), "reviewer": args.reviewer, "notes": args.notes, "license_source": args.license_source, "dataset_fingerprint": report["dataset_fingerprint"]})
    print("Review saved; training can now use this exact dataset version")


if __name__ == "__main__":
    main()
