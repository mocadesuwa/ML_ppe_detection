"""Create a synthetic head-crop workflow example; never use it as training data."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import config_path, project_path, write_json
from src.prepare_head_crops import build_plan, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New directory for synthetic inputs and crops")
    args = parser.parse_args()
    try:
        root = config_path(args.output, "output", project_path("."))
        if root.exists() or root.is_symlink():
            raise FileExistsError("Demo output exists; choose a new directory")
        from PIL import Image, ImageDraw

        source = root / "source" / "synthetic.png"
        source.parent.mkdir(parents=True)
        image = Image.new("RGB", (160, 120), "#d9e3ee")
        draw = ImageDraw.Draw(image)
        draw.ellipse((10, 20, 69, 99), fill="#f1c9a5")
        draw.ellipse((90, 20, 149, 99), fill="#dba982")
        draw.rectangle((18, 43, 61, 59), outline="#225588", width=3)
        image.save(source)
        manifest = root / "head_boxes.json"
        write_json(manifest, {"format_version": 1, "items": [
            {"id": "head_001", "source": "source/synthetic.png", "group": "synthetic_scene", "box_xyxy": [10, 20, 70, 100]},
            {"id": "head_002", "source": "source/synthetic.png", "group": "synthetic_scene", "box_xyxy": [90, 20, 150, 100]},
        ]})
        report = prepare(build_plan(manifest, root / "crops"))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Synthetic demo only: {len(report['crops'])} crops and provenance at {root / 'crops'}")
    print("This example verifies file handling, not goggles labels or model quality.")


if __name__ == "__main__":
    main()
