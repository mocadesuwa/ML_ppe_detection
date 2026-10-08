"""YOLO label boundaries and compatibility with the existing converter."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.check_dataset import read_labels
from src.prepare_dataset import convert_box


class LabelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "labels.txt"

    def parse(self, content, class_count=3):
        self.path.write_text(content, encoding="utf-8")
        return read_labels(self.path, class_count)

    def test_class_count_must_be_a_positive_integer_even_for_empty_labels(self):
        for count in (0, -1, 0.5, True, False, "3", None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.parse("", count)

    def test_individual_normalized_fields_must_stay_in_range(self):
        invalid = (
            "0 -0.00000001 0.5 0.00000002 0.2",
            "0 1.00000001 0.5 0.00000002 0.2",
            "0 0.5 -0.00000001 0.2 0.00000002",
            "0 0.5 1.00000001 0.2 0.00000002",
            "0 0.5 0.5 1.00000001 0.2",
            "0 0.5 0.5 0.2 1.00000001",
            "0 0 0.5 0.000000001 0.2",
            "0 1 0.5 0.000000001 0.2",
            "0 1.00000000000000000001 0.5 0.000000001 0.2",
        )
        for line in invalid:
            with self.subTest(line=line), self.assertRaises(ValueError):
                self.parse(line)

    def test_fractional_class_ids_cannot_round_to_valid_integers(self):
        for category in ("1.00000000000000000001", "0.99999999999999999999", "-1e-400"):
            with self.subTest(category=category), self.assertRaisesRegex(ValueError, "Line 1: invalid class ID"):
                self.parse(f"{category} 0.5 0.5 0.2 0.2")

    def test_numeric_and_field_count_errors_include_line_numbers(self):
        for line in ("0 word 0.5 0.2 0.2", "word 0.5 0.5 0.2 0.2", "0 0.5 0.5 0.2", "0 0.5 0.5 0.2 0.2 extra"):
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, "Line 3:"):
                self.parse("0 0.5 0.5 0.2 0.2\n\n" + line)

    def test_all_non_finite_fields_are_rejected(self):
        for position in range(5):
            for value in ("nan", "inf", "-inf", "1e400"):
                fields = ["0", "0.5", "0.5", "0.2", "0.2"]
                fields[position] = value
                with self.subTest(position=position, value=value), self.assertRaises(ValueError):
                    self.parse(" ".join(fields))

    def test_empty_blank_and_bom_whitespace_labels_remain_supported(self):
        self.assertEqual(self.parse(""), [])
        self.assertEqual(self.parse("\ufeff\n \t\n"), [])
        self.assertEqual(self.parse("\ufeff0\t0.5  0.5\t0.2 0.2\r\n"), [[0.0, 0.5, 0.5, 0.2, 0.2]])

    def test_converted_boundary_boxes_survive_eight_decimal_serialization(self):
        for size in (1, 6, 7, 11, 101, 640, 1024, 7680):
            for convention in ("zero_based_continuous", "voc_one_based_inclusive"):
                boxes = ((0, 0, size, size), (0, 0, 1, 1), (size - 1, size - 1, size, size))
                for box in boxes:
                    source = (box[0] + 1, box[1] + 1, box[2], box[3]) if convention == "voc_one_based_inclusive" else box
                    normalized = convert_box(source, size, size, convention)
                    line = "0 " + " ".join(f"{value:.8f}" for value in normalized)
                    with self.subTest(size=size, convention=convention, box=source):
                        self.assertEqual(len(self.parse(line)), 1)

    def test_overhang_beyond_serialization_tolerance_is_rejected(self):
        for line in ("0 0.09999998 0.5 0.2 0.2", "0 0.90000002 0.5 0.2 0.2", "0 0.5 0.09999998 0.2 0.2", "0 0.5 0.90000002 0.2 0.2"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                self.parse(line)

    def test_zero_negative_and_underflowed_areas_are_rejected(self):
        for width, height in (("0", "0.2"), ("0.2", "0"), ("-0.1", "0.2"), ("0.2", "-0.1"), ("1e-400", "0.2")):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                self.parse(f"0 0.5 0.5 {width} {height}")

    def test_integral_numeric_class_forms_and_multiple_objects_remain_supported(self):
        for category in ("0", "0.0", "0e0", "-0"):
            labels = self.parse(f"{category} 0.5 0.5 1 1\n2 0.25 0.25 0.5 0.5\n")
            self.assertEqual(labels, [[0.0, 0.5, 0.5, 1.0, 1.0], [2.0, 0.25, 0.25, 0.5, 0.5]])
