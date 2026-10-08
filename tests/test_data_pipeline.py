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

    def test_invalid_image_dimensions_are_rejected(self):
        for dimension in (0, -1, 0.5, float("inf"), float("-inf"), float("nan"), True, False, "10", None):
            for width, height in ((dimension, 10), (10, dimension)):
                with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                    convert_box((0, 0, 0.25, 0.25), width, height, "zero_based_continuous")

    def test_inclusive_voc_single_pixel_at_image_edge(self):
        self.assertEqual(convert_box((10, 10, 10, 10), 10, 10, "voc_one_based_inclusive"), (0.95, 0.95, 0.1, 0.1))
        self.assertEqual(convert_box((1, 1, 1, 1), 1, 1, "voc_one_based_inclusive"), (0.5, 0.5, 1.0, 1.0))

    def test_unknown_coordinate_convention_is_rejected(self):
        with self.assertRaises(ValueError):
            convert_box((0, 0, 5, 5), 10, 10, "unknown")


class GroupTests(unittest.TestCase):
    def test_three_independent_groups_cover_all_splits(self):
        for sizes in ((1, 1, 1), (20, 1, 1)):
            samples, groups = [], []
            for size in sizes:
                start = len(samples)
                samples.extend({"labels": [[0, 0.5, 0.5, 0.2, 0.2]]} for _ in range(size))
                groups.append(list(range(start, len(samples))))
            with self.subTest(sizes=sizes):
                assignments = split_groups(groups, samples, [0.7, 0.2, 0.1], 42, 1)
                self.assertEqual({frozenset(indices) for indices in assignments.values()}, {frozenset(group) for group in groups})
                self.assertEqual(assignments, split_groups(groups, samples, [0.7, 0.2, 0.1], 42, 1))

    def test_insufficient_independent_groups_for_a_class_are_rejected(self):
        samples = [{"labels": [[i % 2, 0.5, 0.5, 0.2, 0.2]]} for i in range(4)]
        with self.assertRaises(ValueError):
            split_groups([[i] for i in range(4)], samples, [0.7, 0.2, 0.1], 42, 2)

    def test_group_split_rejects_overlapping_missing_or_invalid_members(self):
        samples = [{"labels": [[0, 0.5, 0.5, 0.2, 0.2]]} for _ in range(9)]
        valid = [[i] for i in range(9)]
        invalid = (
            [[0], [0], *valid[2:]],  # Duplicate 0 also omits 1.
            valid[:-1],
            [*valid, []],
            [[-1], *valid[1:]],
            [[9], *valid[1:]],
            [[0.0], *valid[1:]],
            [[False], *valid[1:]],
            [[0, 0], *valid[1:]],
        )
        for groups in invalid:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                split_groups(groups, samples, [1 / 3] * 3, 42, 1)

    def test_group_split_rejects_invalid_class_ids(self):
        for category in (-1, 1, 0.5, float("nan"), float("inf"), True, False, "0"):
            samples = [{"labels": [[category, 0.5, 0.5, 0.2, 0.2]]} for _ in range(9)]
            with self.subTest(category=category), self.assertRaises(ValueError):
                split_groups([[i] for i in range(9)], samples, [1 / 3] * 3, 42, 1)

    def test_group_split_rejects_invalid_configuration(self):
        samples = [{"labels": [[0, 0.5, 0.5, 0.2, 0.2]]} for _ in range(9)]
        groups = [[i] for i in range(9)]
        for count in (0, -1, 1.5, True, "1"):
            with self.subTest(class_count=count), self.assertRaises(ValueError):
                split_groups(groups, samples, [1 / 3] * 3, 42, count)
        for ratios in ([0.7, 0.2], [0.7, 0.2, 0], [float("nan"), 0.2, 0.1], ["0.7", 0.2, 0.1], [True, 0.2, 0.1]):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                split_groups(groups, samples, ratios, 42, 1)
        for seed in (0.5, True, "42"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                split_groups(groups, samples, [1 / 3] * 3, seed, 1)

    def test_near_duplicate_threshold_and_hash_are_validated(self):
        valid = [{"sha256": "a", "dhash": 0}]
        for distance in (-1, 65, 0.5, True, "1"):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                group_samples(valid, distance)
        for digest in (-1, 1 << 64, 0.5, True, "0"):
            with self.subTest(dhash=digest), self.assertRaises(ValueError):
                group_samples([{"sha256": "a", "dhash": digest}], 0)

    def test_near_duplicate_grouping_is_transitive(self):
        samples = [{"sha256": str(i), "dhash": digest} for i, digest in enumerate((0, 1, 3, (1 << 64) - 1))]
        self.assertEqual(group_samples(samples, 1), [[0, 1, 2], [3]])

    def test_source_csv_groups_and_similarity_are_combined(self):
        samples = [
            {"image": Path(f"image_{i}.png"), "sha256": str(i), "dhash": digest}
            for i, digest in enumerate((0, 1, (1 << 64) - 1, 0xAAAAAAAAAAAAAAAA))
        ]
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "groups.csv"
            csv_path.write_text("filename,group\n image_1.png , session_a \nimage_2.png,session_a\n", encoding="utf-8-sig")
            self.assertEqual(group_samples(samples, 1, csv_path), [[0, 1, 2], [3]])

    def test_malformed_source_csv_is_rejected(self):
        samples = [{"image": Path("a.png"), "sha256": "a", "dhash": 0}]
        invalid_rows = ("a.png", ",group_a", "a.png,group_a,extra", "a.png,group_a\n a.png ,group_b", "unknown.png,group_a")
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "groups.csv"
            for row in invalid_rows:
                csv_path.write_text("filename,group\n" + row + "\n", encoding="utf-8")
                with self.subTest(row=row), self.assertRaises(ValueError):
                    group_samples(samples, 0, csv_path)

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
        self.assertEqual(sum(map(len, first.values())), len(samples))
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
            for member in ("../outside.txt", "..\\outside.txt", "/outside.txt", "C:/outside.txt", "//server/share/outside.txt", "folder/../outside.txt"):
                with zipfile.ZipFile(archive, "w") as zipped:
                    zipped.writestr("valid.txt", "test")
                    zipped.writestr(member, "test")
                with self.subTest(member=member), self.assertRaises(ValueError):
                    extract_archive(archive, root / "destination")
                self.assertFalse((root / "destination").exists())
            self.assertFalse((root / "outside.txt").exists())

    def test_archive_extracts_nested_binary_files_and_normalizes_separators(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "valid.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
                zipped.writestr("images/", b"")
                zipped.writestr("images/sample.bin", b"\x00\xff\r\n")
                zipped.writestr("annotations\\sample.xml", b"<annotation />")
            for existing in (False, True):
                destination = root / f"destination_{existing}"
                if existing:
                    destination.mkdir()
                with self.subTest(existing=existing):
                    extract_archive(archive, destination)
                    self.assertEqual((destination / "images" / "sample.bin").read_bytes(), b"\x00\xff\r\n")
                    self.assertEqual((destination / "annotations" / "sample.xml").read_bytes(), b"<annotation />")
            self.assertEqual({p.name for p in root.iterdir()}, {"valid.zip", "destination_False", "destination_True"})

    def test_archive_rejects_colliding_file_names_before_writing(self):
        import warnings
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, destination = root / "collision.zip", root / "destination"
            for names in (("Image.png", "image.png"), ("folder/a.txt", "folder\\a.txt")):
                with warnings.catch_warnings(), zipfile.ZipFile(archive, "w") as zipped:
                    # Windows ZipInfo normalizes separators while building this
                    # intentionally conflicting archive, triggering this warning.
                    warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
                    for name in names:
                        zipped.writestr(name, name)
                with self.subTest(names=names), self.assertRaises(ValueError):
                    extract_archive(archive, destination)
                self.assertFalse(destination.exists())

    def test_archive_rejects_file_directory_conflicts_before_writing(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, destination = root / "conflict.zip", root / "destination"
            for names in (("folder", "folder/sample.txt"), ("folder/sample.txt", "folder")):
                with zipfile.ZipFile(archive, "w") as zipped:
                    for name in names:
                        zipped.writestr(name, "test")
                with self.subTest(names=names), self.assertRaises(ValueError):
                    extract_archive(archive, destination)
                self.assertFalse(destination.exists())

    def test_archive_does_not_overwrite_existing_data(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, destination = root / "overwrite.zip", root / "destination"
            destination.mkdir()
            original = destination / "sample.txt"
            original.write_text("original", encoding="utf-8")
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("sample.txt", "replacement")
            with self.assertRaises(FileExistsError):
                extract_archive(archive, destination)
            self.assertEqual(original.read_text(encoding="utf-8"), "original")

    def test_archive_rejects_names_that_windows_would_rewrite(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, destination = root / "name.zip", root / "destination"
            for member in ("image?.png", "image.png.", "image.png ", "folder//image.png", "./image.png", "folder/./image.png"):
                with zipfile.ZipFile(archive, "w") as zipped:
                    zipped.writestr(member, "test")
                with self.subTest(member=member), self.assertRaises(ValueError):
                    extract_archive(archive, destination)
                self.assertFalse(destination.exists())

    def test_archive_rejects_links_and_special_files(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, destination = root / "special.zip", root / "destination"
            for mode in (0o120777, 0o010600):
                entry = zipfile.ZipInfo("special")
                entry.create_system = 3
                entry.external_attr = mode << 16
                with zipfile.ZipFile(archive, "w") as zipped:
                    zipped.writestr(entry, "test")
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    extract_archive(archive, destination)
                self.assertFalse(destination.exists())

    def test_archive_crc_failure_leaves_no_partial_dataset(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, destination = root / "corrupt.zip", root / "destination"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zipped:
                zipped.writestr("first.txt", "valid")
                zipped.writestr("second.txt", "CRC-failure-unique")
            content = bytearray(archive.read_bytes())
            content[content.index(b"CRC-failure-unique")] ^= 1
            archive.write_bytes(content)
            with self.assertRaises(zipfile.BadZipFile):
                extract_archive(archive, destination)
            self.assertFalse(destination.exists())
            self.assertEqual({p.name for p in root.iterdir()}, {"corrupt.zip"})

    def test_archive_rejects_empty_dataset(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "empty.zip"
            with zipfile.ZipFile(archive, "w"):
                pass
            with self.assertRaises(ValueError):
                extract_archive(archive, root / "destination")
            self.assertFalse((root / "destination").exists())

    def test_archive_publish_failure_preserves_empty_destination_and_can_retry(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, destination = root / "valid.zip", root / "destination"
            destination.mkdir()
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("sample.txt", "original")
            with patch.object(Path, "rename", side_effect=PermissionError("simulated directory move failure")):
                with self.assertRaises(PermissionError):
                    extract_archive(archive, destination)
            self.assertTrue(destination.is_dir())
            self.assertEqual(list(destination.iterdir()), [])
            self.assertEqual({p.name for p in root.iterdir()}, {"valid.zip", "destination"})
            extract_archive(archive, destination)
            self.assertEqual((destination / "sample.txt").read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
