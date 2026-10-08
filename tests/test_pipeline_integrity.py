"""Synthetic integration tests for frozen data and review provenance."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.check_dataset import validate
from src.common import dataset_config, dataset_fingerprint, require_review, write_json
from src.prepare_dataset import main as prepare_main
from src.review_dataset import main as review_main


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        import yaml
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        raw = self.root / "raw"
        raw.mkdir()
        for index in range(12):
            rng = random.Random(index)
            image = Image.new("L", (16, 16))
            image.putdata([rng.randrange(256) for _ in range(256)])
            image.save(raw / f"sample_{index}.png")
            (raw / f"sample_{index}.xml").write_text(
                f"<annotation><filename>sample_{index}.png</filename><size><width>16</width><height>16</height></size>"
                "<object><name>with_mask</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>15</xmax><ymax>15</ymax></bndbox></object></annotation>", encoding="utf-8")
        self.output = self.root / "prepared"
        self.prepare_path = self.root / "prepare.yaml"
        self.prepare_config = {"raw": str(raw), "output": str(self.output), "seed": 42, "ratios": [0.7, 0.2, 0.1], "coordinates": "zero_based_continuous", "near_duplicate_distance": 0, "classes": {"with_mask": 0}}
        self.prepare_path.write_text(yaml.safe_dump(self.prepare_config), encoding="utf-8")
        self.data_path = self.root / "dataset.yaml"
        self.data_config = {"path": "prepared", "train": "images/train", "val": "images/val", "test": "images/test", "names": ["mask"]}
        self.data_path.write_text(yaml.safe_dump(self.data_config), encoding="utf-8")

    def prepare(self):
        with patch("sys.argv", ["prepare_dataset", "--config", str(self.prepare_path)]), patch("src.prepare_dataset.write_json") as write, contextlib.redirect_stdout(io.StringIO()):
            write.side_effect = lambda path, value: write_json(self.root / "source_check.json" if path.name == "source_check.json" else path, value)
            prepare_main()
        return dataset_config(self.data_path)

    def review(self):
        with patch("sys.argv", ["review_dataset", "--data", str(self.data_path), "--reviewer", "synthetic-test", "--notes", "Synthetic fixtures only", "--license-source", "temporary test fixture"]), contextlib.redirect_stdout(io.StringIO()):
            review_main()

    def test_conversion_rejects_invalid_class_mapping_before_writing(self):
        import yaml
        for classes in ({"with_mask": 0.5}, {"with_mask": False}, {"with_mask": "0"}, {}, {"with_mask": 0, "without_mask": 0}):
            config = dict(self.prepare_config, classes=classes)
            self.prepare_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.output.exists())

    def test_dataset_names_reject_invalid_or_colliding_ids(self):
        import yaml
        for names in ({0.5: "mask"}, {False: "mask"}, {0: "mask", "0": "other"}, [], None, {0: ""}, ["mask", "mask"]):
            self.data_path.write_text(yaml.safe_dump(dict(self.data_config, names=names)), encoding="utf-8")
            with self.subTest(names=names), self.assertRaises(ValueError):
                dataset_config(self.data_path)

    def test_dataset_names_accept_integer_ids_or_digit_strings(self):
        import yaml
        for names in (["mask"], {0: "mask"}, {"0": "mask"}):
            self.data_path.write_text(yaml.safe_dump(dict(self.data_config, names=names)), encoding="utf-8")
            self.assertEqual(dataset_config(self.data_path)["names"], {0: "mask"})

    def test_malformed_manifest_reports_errors_without_crashing(self):
        config = self.prepare()
        manifest = self.output / "manifest.json"
        for content in ("{}", "[]", "null", "{broken", '{"splits": {"train": [{}]}}'):
            manifest.write_text(content, encoding="utf-8")
            with self.subTest(content=content):
                report, _ = validate(config)
                self.assertFalse(report["ok"])
                self.assertTrue(report["errors"])

    def test_manifest_duplicate_and_changed_class_records_are_rejected(self):
        config = self.prepare()
        path = self.output / "manifest.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        duplicate = copy.deepcopy(original)
        duplicate["splits"]["train"].append(duplicate["splits"]["train"][0])
        changed = copy.deepcopy(original)
        changed["splits"]["train"][0]["class_ids"] = []
        for manifest in (duplicate, changed):
            write_json(path, manifest)
            report, _ = validate(config)
            self.assertFalse(report["ok"])

    def test_invalid_manifest_record_fields_are_reported(self):
        config = self.prepare()
        path = self.output / "manifest.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for changes in ({"filename": None}, {"filename": "../sample.png"}, {"group": []}, {"group": True}, {"class_ids": None}, {"class_ids": [0.5]}, {"sha256": "broken"}, {"label_sha256": None}):
            manifest = copy.deepcopy(original)
            manifest["splits"]["train"][0].update(changes)
            write_json(path, manifest)
            with self.subTest(changes=changes):
                report, _ = validate(config)
                self.assertFalse(report["ok"])
        for version in (True, 1.5, 2):
            manifest = dict(original, format_version=version)
            write_json(path, manifest)
            report, _ = validate(config)
            self.assertFalse(report["ok"])

    def test_original_manifest_format_remains_readable(self):
        config = self.prepare()
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest.pop("format_version"), 1)
        for records in manifest["splits"].values():
            for record in records:
                self.assertEqual(len(record.pop("label_sha256")), 64)
        write_json(path, manifest)
        report, _ = validate(config)
        self.assertTrue(report["ok"], report["errors"])
        self.review()
        require_review(config)

    def test_declared_missing_image_does_not_fall_back_to_annotation_stem(self):
        annotation = self.root / "raw" / "sample_0.xml"
        annotation.write_text(annotation.read_text(encoding="utf-8").replace("sample_0.png", "missing.png"), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())
        report = json.loads((self.root / "source_check.json").read_text(encoding="utf-8"))
        self.assertTrue(any("No matching image" in error["error"] for error in report["errors"]))

    def test_missing_filename_still_uses_annotation_stem(self):
        annotation = self.root / "raw" / "sample_0.xml"
        annotation.write_text(annotation.read_text(encoding="utf-8").replace("<filename>sample_0.png</filename>", ""), encoding="utf-8")
        config = self.prepare()
        report, _ = validate(config)
        self.assertTrue(report["ok"], report["errors"])

    def test_removing_manifest_or_adding_orphan_label_invalidates_review(self):
        config = self.prepare()
        self.review()
        path = self.output / "manifest.json"
        content = path.read_bytes()
        path.unlink()
        report, _ = validate(config)
        self.assertFalse(report["ok"])
        with self.assertRaises(RuntimeError):
            require_review(config)
        path.write_bytes(content)
        orphan = self.output / "labels" / "train" / "orphan.txt"
        orphan.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        report, _ = validate(config)
        self.assertFalse(report["ok"])
        with self.assertRaises(RuntimeError):
            require_review(config)

    def test_changed_label_geometry_disagrees_with_frozen_manifest(self):
        config = self.prepare()
        label = next((self.output / "labels" / "train").glob("*.txt"))
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        report, _ = validate(config)
        self.assertFalse(report["ok"])
        with self.assertRaises(ValueError):
            self.review()

    def test_changed_split_list_is_reported_and_invalidates_review(self):
        config = self.prepare()
        self.review()
        require_review(config)
        path = self.output / "splits" / "train.txt"
        path.write_text("unexpected.png\n", encoding="utf-8")
        report, _ = validate(config)
        self.assertFalse(report["ok"])
        with self.assertRaises(RuntimeError):
            require_review(config)

    def test_changed_manifest_invalidates_review(self):
        config = self.prepare()
        self.review()
        before = dataset_fingerprint(config)
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["splits"]["train"][0]["group"] += 1000
        write_json(path, manifest)
        self.assertNotEqual(before, dataset_fingerprint(config))
        with self.assertRaises(RuntimeError):
            require_review(config)

    def test_bad_review_record_has_clear_error(self):
        config = self.prepare()
        path = self.output / "quality_review.json"
        for content in ("[]", "null", "{broken"):
            path.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(RuntimeError):
                require_review(config)

    def test_review_tracks_image_label_names_and_membership_changes(self):
        config = self.prepare()
        self.review()
        baseline = dataset_fingerprint(config)
        image = next((self.output / "images" / "train").glob("*.png"))
        label = self.output / "labels" / "train" / (image.stem + ".txt")
        for path, replacement in ((image, image.read_bytes() + b"changed"), (label, b"0 0.5 0.5 0.2 0.2\n")):
            original = path.read_bytes()
            path.write_bytes(replacement)
            self.assertNotEqual(baseline, dataset_fingerprint(config))
            with self.assertRaises(RuntimeError):
                require_review(config)
            path.write_bytes(original)
        changed = dict(config, names={0: "different"})
        with self.assertRaises(RuntimeError):
            require_review(changed)
        destination = self.output / "images" / "val" / image.name
        image.rename(destination)
        try:
            with self.assertRaises(RuntimeError):
                require_review(config)
        finally:
            destination.rename(image)
        self.assertEqual(baseline, dataset_fingerprint(config))
        require_review(config)
