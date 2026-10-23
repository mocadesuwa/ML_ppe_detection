"""Classifier metrics, weighted-loss regression and final-test provenance checks."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.classification_model import (MODEL, PREPROCESSING, classification_metrics, completed_training_record,
                                      prediction_row, validate_checkpoint_metadata)
from src.common import sha256, write_json
from src.evaluate_classification import build_plan, independent_benchmark
from src.predict_classification import run as predict
from src.train_classification import source_group_sampling_weights, weighted_loss_totals


class ClassificationMetricsTests(unittest.TestCase):
    def test_class_source_sampler_preserves_both_classes_and_balances_groups_within_each(self):
        rows = [{"id": f"negative_{i}", "group": "mixed", "label": "class_0"} for i in range(120)]
        rows += [{"id": f"positive_{i}", "group": "mixed", "label": "class_1"} for i in range(32)]
        rows += [{"id": "other_negative", "group": "other", "label": "class_0"}]
        rows += [{"id": "other_positive_a", "group": "a", "label": "class_1"},
                 {"id": "other_positive_b", "group": "b", "label": "class_1"}]
        ids = [row["id"] for row in reversed(rows)]
        weights = dict(zip(ids, source_group_sampling_weights(ids, rows, balance_classes=True)))
        for label, group_count in (("class_0", 2), ("class_1", 3)):
            labelled = [row for row in rows if row["label"] == label]
            self.assertAlmostEqual(sum(weights[row["id"]] for row in labelled), 1)
            for group in {row["group"] for row in labelled}:
                self.assertAlmostEqual(sum(weights[row["id"]] for row in labelled if row["group"] == group), 1 / group_count)
        self.assertEqual(weights["other_positive_a"], weights["other_positive_b"])

    def test_class_source_sampler_rejects_missing_labels_and_single_class(self):
        for rows in ([{"id": "a", "group": "g"}],
                     [{"id": "a", "group": "g", "label": ""}],
                     [{"id": "a", "group": "g", "label": "class_0"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                source_group_sampling_weights(["a"], rows, balance_classes=True)
        with self.assertRaises(ValueError):
            source_group_sampling_weights(["a", "a"], [{"id": "a", "group": "g", "label": "class_0"}], balance_classes=True)

    def test_source_groups_have_equal_sampling_mass_despite_repeated_frames(self):
        rows = [{"id": f"frame_{i}", "group": "factory"} for i in range(120)]
        rows += [{"id": "independent_a", "group": "a"}, {"id": "independent_b", "group": "b"}]
        ids = [row["id"] for row in reversed(rows)]
        weights = source_group_sampling_weights(ids, rows)
        by_id = dict(zip(ids, weights))
        self.assertAlmostEqual(sum(by_id[f"frame_{i}"] for i in range(120)), 1)
        self.assertEqual(by_id["independent_a"], 1)
        self.assertEqual(by_id["independent_b"], 1)

    def test_source_sampler_rejects_missing_duplicate_or_unknown_ids(self):
        rows = [{"id": "a", "group": "group_a"}, {"id": "b", "group": "group_b"}]
        for ids in (["a"], ["a", "a"], ["a", "other"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                source_group_sampling_weights(ids, rows)

    def test_imbalanced_all_negative_baseline_has_zero_positive_recall(self):
        result = classification_metrics([[24, 0], [5, 0]], ["class_0", "class_1"])
        self.assertAlmostEqual(result["accuracy"], 24 / 29)
        self.assertEqual(result["per_class"][1]["recall"], 0)
        self.assertEqual(result["per_class"][1]["precision"], 0)
        self.assertAlmostEqual(result["macro_f1"], (48 / 53) / 2)

    def test_perfect_and_mixed_confusion_metrics(self):
        perfect = classification_metrics([[3, 0], [0, 2]], ["class_0", "class_1"])
        self.assertEqual(perfect["accuracy"], 1)
        self.assertEqual(perfect["macro_f1"], 1)
        mixed = classification_metrics([[3, 1], [2, 4]], ["class_0", "class_1"])
        self.assertAlmostEqual(mixed["per_class"][1]["precision"], 4 / 5)
        self.assertAlmostEqual(mixed["per_class"][1]["recall"], 4 / 6)
        self.assertEqual(mixed["image_count"], 10)

    def test_invalid_or_empty_confusion_rejected(self):
        for matrix in ([[0, 0], [0, 0]], [[1, -1], [0, 1]], [[True, 0], [0, 1]], [[1], [2]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                classification_metrics(matrix, ["class_0", "class_1"])

    def test_weighted_epoch_loss_does_not_depend_on_batch_partition(self):
        losses, weights = [-math.log(value) for value in [.8, .6, .1]], [.5, .5, 2]
        expected = sum(loss * weight for loss, weight in zip(losses, weights)) / sum(weights)
        for partition in ([[0, 1], [2]], [[0], [1, 2]], [[0, 1, 2]]):
            numerator, denominator = 0.0, 0.0
            for batch in partition:
                weight = sum(weights[index] for index in batch)
                loss = sum(losses[index] * weights[index] for index in batch) / weight
                numerator, denominator = weighted_loss_totals(loss, weight, numerator, denominator)
            self.assertAlmostEqual(numerator / denominator, expected)
        wrong_old_mean = ((losses[0] + losses[1]) / 2 * 2 + losses[2]) / 3
        self.assertNotAlmostEqual(wrong_old_mean, expected)

    def test_invalid_weighted_loss_rejected(self):
        for loss, weight in ((float("nan"), 1), (1, 0), (1, float("inf"))):
            with self.subTest(loss=loss, weight=weight), self.assertRaises(ValueError):
                weighted_loss_totals(loss, weight, 0, 0)

    def test_prediction_probabilities_and_tie_rule(self):
        self.assertEqual(prediction_row([.3, .7], ["class_0", "class_1"])["label"], "class_1")
        self.assertEqual(prediction_row([.5, .5], ["class_0", "class_1"])["class_id"], 0)
        for probabilities in ([.3, .8], [float("nan"), .5], [-.1, 1.1], [1], [True, 0]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                prediction_row(probabilities, ["class_0", "class_1"])

    def test_checkpoint_contract_rejects_task_preprocessing_and_class_drift(self):
        checkpoint = {"format_version": 1, "task": "classification", "model": MODEL,
                      "imgsz": 224, "preprocessing": PREPROCESSING, "classes": ["class_0", "class_1"],
                      "state_dict": {}}
        self.assertEqual(validate_checkpoint_metadata(checkpoint), ["class_0", "class_1"])
        for key, value in (("task", "detect"), ("imgsz", 640), ("preprocessing", "center_crop"),
                           ("classes", ["class_1", "class_0"]), ("classes", ["class_0", "class_0"]),
                           ("state_dict", None)):
            changed = copy.deepcopy(checkpoint)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_checkpoint_metadata(changed)
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata(checkpoint, ["other_0", "other_1"])


class ClassificationProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.weights = self.root / "best.pt"
        self.weights.write_bytes(b"metadata-only fixture; never loaded as a model")
        self.config = self.root / "plan.yaml"
        self.config.write_text("fixture: true\n", encoding="utf-8")
        self.record = {"task": "classification", "status": "complete", "mode": "baseline",
                       "best_weights_sha256": sha256(self.weights), "checkpoint_reload_verified": True,
                       "frozen_features_unchanged": True, "classifier_updated": True,
                       "config_sha256": sha256(self.config), "dataset_fingerprint": "f" * 64,
                       "parameters": {"classes": ["class_0", "class_1"]}}
        self.save_record()
        self.mock_plan = patch("src.evaluate_classification.training_plan", return_value=(
            {"data": str(self.root / "data")}, {"dataset_fingerprint": "f" * 64}))
        self.mock_plan.start()
        self.addCleanup(self.mock_plan.stop)

    def save_record(self):
        write_json(self.root / "experiment.json", self.record)

    def plan(self, split="val", final=False):
        return build_plan(self.config, self.weights, split, final, "fixture_workflow")

    def test_completed_baseline_can_plan_explicit_final_test(self):
        plan, _ = self.plan("test", True)
        self.assertTrue(plan["final_test"])
        self.assertFalse((self.root / "final_test.json").exists())

    def test_final_test_requires_flag_and_test_split(self):
        for split, final in (("test", False), ("val", True), ("train", False)):
            with self.subTest(split=split, final=final), self.assertRaises(ValueError):
                self.plan(split, final)

    def test_smoke_can_validate_but_cannot_final_test(self):
        self.record["mode"] = "smoke"
        self.save_record()
        self.assertEqual(self.plan()[0]["training_mode"], "smoke")
        with self.assertRaisesRegex(ValueError, "not a smoke"):
            self.plan("test", True)

    def test_incomplete_or_unverified_training_is_rejected(self):
        for key, value in (("status", "failed"), ("checkpoint_reload_verified", False),
                           ("frozen_features_unchanged", False), ("classifier_updated", False)):
            original = copy.deepcopy(self.record)
            self.record[key] = value
            self.save_record()
            with self.subTest(key=key), self.assertRaises(ValueError):
                completed_training_record(self.weights)
            self.record = original

    def test_changed_checkpoint_data_or_config_is_rejected(self):
        original = self.weights.read_bytes()
        self.weights.write_bytes(b"changed weights")
        with self.assertRaises(ValueError):
            self.plan()
        self.weights.write_bytes(original)
        self.record["dataset_fingerprint"] = "0" * 64
        self.save_record()
        with self.assertRaisesRegex(ValueError, "config or data"):
            self.plan()
        self.record["dataset_fingerprint"] = "f" * 64
        self.save_record()
        self.config.write_text("changed: true\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "config or data"):
            self.plan()

    def test_final_test_cannot_repeat_after_any_attempt(self):
        for status in ("running", "failed", "complete"):
            write_json(self.root / "final_test.json", {"status": status})
            with self.subTest(status=status), self.assertRaises(FileExistsError):
                self.plan("test", True)

    def test_external_benchmark_cannot_reset_same_dataset_final_test(self):
        with self.assertRaisesRegex(ValueError, "new dataset"):
            build_plan(self.config, self.weights, "test", True, "fixture_workflow", benchmark=True)

    def test_new_independent_benchmark_preserves_old_test_marker(self):
        old_marker = self.root / "final_test.json"
        write_json(old_marker, {"status": "complete", "historical": True})
        historical = old_marker.read_bytes()
        self.record["parameters"]["data"] = str(self.root / "old_data")
        self.save_record()
        with patch("src.evaluate_classification.training_plan", return_value=(
                {"data": str(self.root / "new_data")}, {"dataset_fingerprint": "e" * 64})), \
             patch("src.evaluate_classification.independent_benchmark", return_value={
                 "reference_dataset_fingerprint": "f" * 64, "known_overlap_count": 0}) as independent:
            plan, _ = build_plan(self.config, self.weights, "test", True, "fixture_workflow", benchmark=True)
            self.assertEqual(Path(plan["marker"]).name, "benchmark_" + "e" * 64 + ".json")
            self.assertEqual(plan["trained_dataset_fingerprint"], "f" * 64)
            independent.assert_called_once()
            write_json(Path(plan["marker"]), {"status": "complete"})
            with self.assertRaises(FileExistsError):
                build_plan(self.config, self.weights, "test", True, "fixture_workflow", benchmark=True)
        self.assertEqual(old_marker.read_bytes(), historical)

    def test_smoke_rejected_before_external_benchmark_inspection(self):
        self.record["mode"] = "smoke"
        self.save_record()
        with patch("src.evaluate_classification.independent_benchmark") as independent:
            with self.assertRaisesRegex(ValueError, "not a smoke"):
                build_plan(self.config, self.weights, "test", True, "fixture_workflow", benchmark=True)
            independent.assert_not_called()

    def test_prediction_rejects_invalid_static_source_before_model_loading(self):
        source = self.root / "animated.webp"
        Image.new("RGB", (8, 8), "red").save(source, save_all=True,
            append_images=[Image.new("RGB", (8, 8), "blue")], duration=100, loop=0)
        with patch("src.predict_classification.load_classifier") as load:
            with self.assertRaisesRegex(ValueError, "static"):
                predict(self.weights, source, "fixture_invalid")
            load.assert_not_called()


class IndependentBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old, self.new = self.root / "old", self.root / "new"
        self.old.mkdir(); self.new.mkdir()
        self.old_rows = [{"id": name, "group": "group_" + name, "sha256": "png_" + name,
                          "pixel_sha256": "pixels_" + name} for name in ("train", "val", "test")]
        self.new_row = {"id": "new", "group": "group_new", "sha256": "png_new", "pixel_sha256": "pixels_new"}
        self.old_manifest = {"classes": ["class_0", "class_1"], "crop_directory": str(self.old),
                             "splits": {name: [row] for name, row in zip(("train", "val", "test"), self.old_rows)}}
        self.new_manifest = {"classes": ["class_0", "class_1"], "crop_directory": str(self.new),
                             "splits": {"test": [self.new_row]}}
        write_json(self.old / "manifest.json", self.old_manifest)
        write_json(self.old / "crops_manifest.json", {"crops": [dict(row, source_sha256="raw_" + row["id"]) for row in self.old_rows]})
        write_json(self.new / "crops_manifest.json", {"crops": [dict(self.new_row, source_sha256="raw_new")]})
        self.mock = patch("src.evaluate_classification.inspect_dataset", return_value={"dataset_fingerprint": "f" * 64})
        self.mock.start(); self.addCleanup(self.mock.stop)

    def check(self):
        write_json(self.new / "manifest.json", self.new_manifest)
        return independent_benchmark(self.new, self.old)

    def test_disjoint_benchmark_excludes_all_old_splits(self):
        result = self.check()
        self.assertEqual(result["excluded_old_images"], 3)
        self.assertEqual(result["known_overlap_count"], 0)
        self.assertEqual(len(result["metadata_sha256"]), 3)

    def test_old_test_ids_groups_or_duplicate_pixels_are_not_new_benchmarks(self):
        for field in ("id", "group", "sha256", "pixel_sha256"):
            original = self.new_row[field]
            self.new_row[field] = self.old_rows[2][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "overlaps"):
                self.check()
            self.new_row[field] = original

    def test_different_crops_from_same_old_original_are_rejected(self):
        write_json(self.new / "crops_manifest.json", {"crops": [dict(self.new_row, source_sha256="raw_test")]})
        with self.assertRaisesRegex(ValueError, "original image bytes"):
            self.check()


if __name__ == "__main__":
    unittest.main()
