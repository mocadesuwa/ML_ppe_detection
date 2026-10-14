"""Explicit head cropping and provenance, using synthetic images only."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common import sha256, write_json
from src.prepare_head_crops import build_plan, main, prepare


class HeadCropTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "输入图片" / "scene.png"
        self.source.parent.mkdir()
        image = Image.new("RGB", (8, 6))
        image.putdata([(x * 20, y * 30, 60) for y in range(6) for x in range(8)])
        image.save(self.source)
        self.original = self.source.read_bytes()
        self.manifest = self.root / "heads.json"
        self.output = self.root / "prepared"
        self.item = {"id": "head_001", "source": "输入图片/scene.png", "group": "source_group", "box_xyxy": [1, 1, 5, 4]}

    def plan(self, items=None):
        write_json(self.manifest, {"format_version": 1, "items": [self.item] if items is None else items})
        return build_plan(self.manifest, self.output)

    def test_pixels_geometry_labels_and_source_groups_are_traceable(self):
        from PIL import Image

        second = {**self.item, "id": "head_002", "box_xyxy": [4, 0, 8, 3]}
        report = prepare(self.plan([{**self.item, "label": "unreviewed"}, second]))
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(report["source_manifest_sha256"], sha256(self.manifest))
        self.assertEqual(len(report["sources"]), 1)
        self.assertEqual([item["group"] for item in report["crops"]], ["source_group", "source_group"])
        self.assertEqual(report["crops"][0]["label"], "unreviewed")
        self.assertNotIn("label", report["crops"][1])
        self.assertFalse(report["visual_review_completed"])
        self.assertEqual(report["status"], "complete")
        for item, size, pixel in zip(report["crops"], [(4, 3), (4, 3)], [(20, 30, 60), (80, 0, 60)]):
            path = self.output / item["output_file"]
            self.assertEqual(item["source_sha256"], sha256(self.source))
            self.assertEqual(item["output_sha256"], sha256(path))
            with Image.open(path) as crop:
                self.assertEqual(crop.size, size)
                self.assertEqual(crop.mode, "RGB")
                self.assertEqual(crop.getpixel((0, 0)), pixel)
        self.assertEqual(json.loads((self.output / "crops_manifest.json").read_text(encoding="utf-8")), report)

    def test_dry_run_checks_inputs_without_creating_output(self):
        self.plan()
        before = sorted(str(path) for path in self.root.rglob("*"))
        stdout = io.StringIO()
        with patch("sys.argv", ["crop", "--manifest", str(self.manifest), "--output", str(self.output), "--dry-run"]), contextlib.redirect_stdout(stdout):
            main()
        self.assertEqual(len(json.loads(stdout.getvalue())["items"]), 1)
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob("*")))

    def test_cli_publishes_synthetic_crops_without_model_dependencies(self):
        self.plan()
        with patch("sys.argv", ["crop", "--manifest", str(self.manifest), "--output", str(self.output)]), patch.dict("sys.modules", {"torch": None, "ultralytics": None}), contextlib.redirect_stdout(io.StringIO()):
            main()
        self.assertTrue((self.output / "images/head_001.png").is_file())

    def test_invalid_boxes_are_rejected_without_clamping_or_outputs(self):
        for box in ([-1, 0, 4, 3], [0, 0, 9, 3], [0, 0, 4, 7], [2, 2, 2, 4], [4, 2, 1, 3], [0.0, 0, 4, 3], [False, 0, 4, 3], [0, 0, 1], "box"):
            with self.subTest(box=box), self.assertRaises(ValueError):
                self.plan([{**self.item, "box_xyxy": box}])
        self.assertFalse(self.output.exists())

    def test_ids_cannot_collide_or_escape_output(self):
        for identifier in ("HEAD_001", "../escape", "a/b", "CON", "", True):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                self.plan([self.item, {**self.item, "id": identifier}])
        self.assertFalse(self.output.exists())

    def test_groups_and_optional_labels_require_explicit_text(self):
        for field in ("group", "label"):
            for value in (None, "", " ", False, 1):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.plan([{**self.item, field: value}])

    def test_crops_of_identical_content_cannot_use_different_groups(self):
        copied = self.source.with_name("copy.png")
        copied.write_bytes(self.original)
        for source in (self.item["source"], "输入图片/copy.png"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "same group"):
                self.plan([self.item, {**self.item, "id": "head_002", "source": source, "group": "different"}])

    def test_manifest_schema_and_duplicate_fields_are_rejected(self):
        for data in ({"format_version": True, "items": [self.item]}, {"format_version": 2, "items": [self.item]}, {"format_version": 1, "items": []}, {"format_version": 1, "items": [{**self.item, "box": [0, 0, 1, 1]}]}, {"items": [self.item]}, []):
            write_json(self.manifest, data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                build_plan(self.manifest, self.output)
        self.manifest.write_text('{"format_version": 1, "format_version": 1, "items": []}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Repeated"):
            build_plan(self.manifest, self.output)

    def test_existing_output_and_source_directory_are_protected(self):
        self.plan()
        for output in (self.source.parent, self.source.parent / "crops"):
            with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                build_plan(self.manifest, output)
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            build_plan(self.manifest, self.output)
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_changed_sources_or_manifest_are_not_published(self):
        plan = self.plan()
        self.source.write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "Source image changed"):
            prepare(plan)
        self.source.write_bytes(self.original)
        self.manifest.write_text(self.manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            prepare(plan)
        self.assertFalse(self.output.exists())

    def test_failed_crop_save_leaves_no_published_directory(self):
        from PIL import Image

        plan = self.plan()
        with patch.object(Image.Image, "save", side_effect=OSError("Synthetic save failure")), self.assertRaisesRegex(OSError, "Synthetic"):
            prepare(plan)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".head-crops-*")), [])
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_animated_corrupt_and_oriented_images_require_review(self):
        from PIL import Image

        animation = self.source.with_name("animated.png")
        first, second = Image.new("RGB", (8, 6), "red"), Image.new("RGB", (8, 6), "blue")
        first.save(animation, save_all=True, append_images=[second], duration=100, loop=0)
        oriented = self.source.with_name("oriented.jpg")
        exif = Image.Exif()
        exif[274] = 6
        first.save(oriented, exif=exif)
        corrupt = self.source.with_name("broken.png")
        corrupt.write_bytes(b"not an image")
        for source in (animation, oriented, corrupt):
            with self.subTest(source=source), self.assertRaises((ValueError, OSError)):
                self.plan([{**self.item, "source": str(source)}])
        self.assertFalse(self.output.exists())
