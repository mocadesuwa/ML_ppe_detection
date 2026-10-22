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
from src.evaluate_classification import build_plan
from src.predict_classification import run as predict
from src.train_classification import weighted_loss_totals


class ClassificationMetricsTests(unittest.TestCase):
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

    def test_prediction_rejects_invalid_static_source_before_model_loading(self):
        source = self.root / "animated.webp"
        Image.new("RGB", (8, 8), "red").save(source, save_all=True,
            append_images=[Image.new("RGB", (8, 8), "blue")], duration=100, loop=0)
        with patch("src.predict_classification.load_classifier") as load:
            with self.assertRaisesRegex(ValueError, "static"):
                predict(self.weights, source, "fixture_invalid")
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
