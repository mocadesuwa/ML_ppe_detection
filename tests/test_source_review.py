"""Read-only source audit evidence and output protection on synthetic samples."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.review_raw_dataset import inspect, main, snapshot


class SourceReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        (self.raw / "images").mkdir(parents=True)
        (self.raw / "annotations").mkdir()

    def sample(self, name="sample", boxes=None, size=(10, 12), class_name="with_mask"):
        from PIL import Image

        Image.new("RGB", (10, 12), (50, 70, 90)).save(self.raw / "images" / f"{name}.png")
        objects = []
        for box in boxes if boxes is not None else [[1, 2, 6, 7]]:
            fields = "".join(f"<{key}>{value}</{key}>" for key, value in zip(("xmin", "ymin", "xmax", "ymax"), box))
            objects.append(f"<object><name>{class_name}</name><bndbox>{fields}</bndbox></object>")
        text = f"<annotation><filename>{name}.png</filename><size><width>{size[0]}</width><height>{size[1]}</height></size>{''.join(objects)}</annotation>"
        (self.raw / "annotations" / f"{name}.xml").write_text(text, encoding="utf-8")

    def test_all_invalid_objects_are_reported_without_changing_source(self):
        self.sample(boxes=[[1, 2, 11, 7], [5, 3, 11, 8], [1, 1, 4, 4]])
        before = snapshot(self.raw)
        report = inspect(self.raw)
        self.assertEqual([obj["index"] for obj in report["invalid_boxes"]], [1, 2])
        self.assertEqual(report["object_count"], 3)
        self.assertEqual(report["geometry"]["zero_based_continuous_valid_objects"], 1)
        self.assertEqual(snapshot(self.raw), before)

    def test_conventions_are_compared_without_guessing_a_winner(self):
        self.sample(boxes=[[0, 0, 4, 4], [5, 5, 5, 5]])
        report = inspect(self.raw)
        self.assertTrue(report["invalid_boxes"][0]["zero_based_continuous"])
        self.assertFalse(report["invalid_boxes"][0]["voc_one_based_inclusive"])
        self.assertFalse(report["invalid_boxes"][1]["zero_based_continuous"])
        self.assertTrue(report["invalid_boxes"][1]["voc_one_based_inclusive"])

    def test_missing_annotations_and_size_mismatch_are_file_errors(self):
        self.sample("wrong_size", size=(11, 12))
        self.sample("unpaired")
        (self.raw / "annotations/unpaired.xml").unlink()
        report = inspect(self.raw)
        self.assertEqual(len(report["file_errors"]), 2)
        self.assertTrue(any("size" in e["error"] for e in report["file_errors"]))
        self.assertTrue(any("Missing annotation" in e["error"] for e in report["file_errors"]))

    def test_identical_images_with_different_box_labels_are_not_silently_merged(self):
        self.sample("first")
        self.sample("second", boxes=[[2, 2, 7, 7]])
        report = inspect(self.raw)
        self.assertEqual(report["exact_duplicate_groups"], [{"filenames": ["first.png", "second.png"], "conflicting_labels": True}])
        self.assertEqual(len(report["records"]), 2)

    def test_source_fingerprint_binds_annotation_content(self):
        self.sample()
        before = snapshot(self.raw)
        path = self.raw / "annotations/sample.xml"
        path.write_text(path.read_text(encoding="utf-8").replace("with_mask", "without_mask"), encoding="utf-8")
        self.assertNotEqual(snapshot(self.raw)[1], before[1])

    def test_output_cannot_overwrite_an_audit_or_enter_raw_directory(self):
        self.sample()
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "keep.txt").write_text("keep", encoding="utf-8")
        before = snapshot(self.raw)
        for output in (existing, self.raw, self.raw / "new"):
            with self.subTest(output=output), patch("sys.argv", ["audit", "--raw", str(self.raw), "--output", str(output)]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
        self.assertEqual((existing / "keep.txt").read_text(encoding="utf-8"), "keep")
        self.assertEqual(snapshot(self.raw), before)


if __name__ == "__main__":
    unittest.main()
