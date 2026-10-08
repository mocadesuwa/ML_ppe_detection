"""Regression checks for annotation geometry, split leakage and provenance."""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.check_dataset import read_labels, validate
from src.common import dataset_config, dataset_fingerprint, require_review, write_json
from src.download_data import extract_archive
from src.prepare_dataset import convert_box, group_samples, main as prepare_main, split_groups


class BoxTests(unittest.TestCase):
    def test_continuous_coordinates(self):
        self.assertEqual(convert_box((200, 150, 400, 450), 800, 600, "zero_based_continuous"), (0.375, 0.5, 0.25, 0.5))

    def test_inclusive_voc_full_image(self):
        self.assertEqual(convert_box((1, 1, 800, 600), 800, 600, "voc_one_based_inclusive"), (0.5, 0.5, 1.0, 1.0))

    def test_invalid_boxes_are_rejected(self):
        for box in ((-1, 0, 5, 5), (2, 2, 2, 3), (0, 0, 11, 5), (0, 0, float("nan"), 5)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                convert_box(box, 10, 10, "zero_based_continuous")


class GroupTests(unittest.TestCase):
    def test_near_duplicates_and_exact_duplicates_share_a_group(self):
        samples = [{"sha256": "a", "dhash": 0}, {"sha256": "b", "dhash": 1}, {"sha256": "c", "dhash": (1 << 64) - 1}, {"sha256": "a", "dhash": 123}]
        groups = [set(group) for group in group_samples(samples, 1)]
        self.assertIn({0, 1, 3}, groups)
        self.assertIn({2}, groups)

    def test_group_split_is_deterministic_and_contains_each_class(self):
        samples = [{"labels": [[i % 3, 0.5, 0.5, 0.2, 0.2]]} for i in range(60)]
        groups = [list(range(i, i + 3)) for i in range(0, 60, 3)]
        first = split_groups(groups, samples, [0.7, 0.2, 0.1], 42, 3)
        self.assertEqual(first, split_groups(groups, samples, [0.7, 0.2, 0.1], 42, 3))
        membership = {index: split for split, indices in first.items() for index in indices}
        self.assertEqual(len(membership), len(samples))
        for group in groups:
            self.assertEqual(len({membership[i] for i in group}), 1)
        for indices in first.values():
            self.assertEqual({int(label[0]) for i in indices for label in samples[i]["labels"]}, {0, 1, 2})


class PipelineTests(unittest.TestCase):
    def test_conversion_validation_and_changed_data_invalidates_review(self):
        from PIL import Image
        import yaml
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, output = root / "raw", root / "prepared"
            raw.mkdir()
            names = ["with_mask", "without_mask", "mask_weared_incorrect"]
            for i in range(30):
                rng = random.Random(i)
                image = Image.new("L", (16, 16))
                image.putdata([rng.randrange(256) for _ in range(256)])
                image.save(raw / f"sample_{i}.png")
                (raw / f"sample_{i}.xml").write_text(f"<annotation><filename>sample_{i}.png</filename><size><width>16</width><height>16</height></size><object><name>{names[i % 3]}</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>15</xmax><ymax>15</ymax></bndbox></object></annotation>", encoding="utf-8")
            config_path = root / "prepare.yaml"
            config_path.write_text(yaml.safe_dump({"raw": str(raw), "output": str(output), "seed": 42, "ratios": [0.7, 0.2, 0.1], "coordinates": "zero_based_continuous", "near_duplicate_distance": 0, "classes": {name: index for index, name in enumerate(names)}}), encoding="utf-8")
            with patch("sys.argv", ["prepare_dataset", "--config", str(config_path)]), patch("src.prepare_dataset.write_json") as write:
                # Keep source-check output in the temporary test directory.
                write.side_effect = lambda path, value: write_json(root / "source_check.json" if path.name == "source_check.json" else path, value)
                prepare_main()
            data_path = root / "dataset.yaml"
            data_path.write_text(yaml.safe_dump({"path": "prepared", "train": "images/train", "val": "images/val", "test": "images/test", "names": ["mask", "no_mask", "mask_incorrect"]}), encoding="utf-8")
            config = dataset_config(data_path)
            report, _ = validate(config)
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(sum(split["images"] for split in report["splits"].values()), 30)
            write_json(output / "quality_review.json", {"dataset_fingerprint": dataset_fingerprint(config)})
            require_review(config)
            label = next((output / "labels" / "train").glob("*.txt"))
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                require_review(config)
            with patch("sys.argv", ["prepare_dataset", "--config", str(config_path)]), self.assertRaises(FileExistsError):
                prepare_main()

    def test_label_fields_and_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.txt"
            for line in ("0 0.9 0.5 0.5 0.2", "1.2 0.5 0.5 0.2 0.2", "3 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 0.5"):
                path.write_text(line, encoding="utf-8")
                with self.subTest(line=line), self.assertRaises(ValueError):
                    read_labels(path, 3)
            path.write_text("0 0.5 0.5 1 1\n", encoding="utf-8")
            self.assertEqual(len(read_labels(path, 3)), 1)

    def test_archive_cannot_extract_outside_destination(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("../outside.txt", "test")
            with self.assertRaises(ValueError):
                extract_archive(archive, root / "destination")
            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
