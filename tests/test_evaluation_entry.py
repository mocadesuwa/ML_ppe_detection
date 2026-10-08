"""Evaluation control-flow tests using synthetic data and a fake YOLO backend."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.common import dataset_config, dataset_fingerprint, sha256, write_json
from src.evaluate import main as evaluate_main


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        import yaml

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        base = self.root / "dataset"
        for index, split in enumerate(("train", "val", "test")):
            images, labels = base / "images" / split, base / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (8, 8), (index * 50, 20, 30)).save(images / f"sample_{index}.png")
            (labels / f"sample_{index}.txt").write_text("0 0.25 0.25 0.5 0.5\n1 0.75 0.75 0.5 0.5\n", encoding="utf-8")
        self.data_path = self.root / "dataset.yaml"
        self.data_path.write_text(yaml.safe_dump({"path": str(base), "train": "images/train", "val": "images/val", "test": "images/test", "names": ["mask", "no_mask"]}), encoding="utf-8")
        self.data = dataset_config(self.data_path)
        self.fingerprint = dataset_fingerprint(self.data)
        self.review_path = base / "quality_review.json"
        write_json(self.review_path, {"dataset_fingerprint": self.fingerprint, "notes": "Synthetic test fixture only"})
        self.weights = self.root / "baseline" / "weights" / "best.pt"
        self.weights.parent.mkdir(parents=True)
        self.weights.write_bytes(b"synthetic fixture, not model weights")
        self.last = self.weights.with_name("last.pt")
        self.last.write_bytes(b"another synthetic fixture")
        self.experiment_path = self.weights.parent.parent / "experiment.json"
        self.record = {"mode": "baseline", "status": "complete", "dataset_fingerprint": self.fingerprint, "best_weights": str(self.weights), "last_weights": str(self.last)}
        write_json(self.experiment_path, self.record)
        box = types.SimpleNamespace(ap_class_index=[1, 0], p=[0.8, 0.6], r=[0.7, 0.5], ap50=[0.9, 0.4], ap=[0.3, 0.2], mp=0.7, mr=0.6, map50=0.65, map=0.25)
        self.metrics = types.SimpleNamespace(box=box, speed={"inference": 1.0})
        self.model = types.SimpleNamespace(names={0: "mask", 1: "no_mask"}, val=Mock(return_value=self.metrics))
        self.yolo = Mock(return_value=self.model)
        self.backend = types.ModuleType("ultralytics")
        self.backend.YOLO = self.yolo
        self.caches, self.device = Mock(), Mock(return_value="cpu")
        self.outputs = self.root / "results" / "evaluations"

    def invoke(self, extra=(), backend=True):
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        argv = ["evaluate", "--weights", str(self.weights), "--data", str(self.data_path), "--device", "cpu", *extra]
        with patch("sys.argv", argv), patch("src.common.ROOT", self.root), patch.dict("sys.modules", {"ultralytics": self.backend if backend else None}), patch("src.evaluate.local_caches", self.caches), patch("src.evaluate.select_device", self.device), contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            evaluate_main()

    def rejected(self, extra=(), message=None, before_model=True):
        with self.assertRaises(SystemExit) as error:
            self.invoke(extra)
        self.assertEqual(error.exception.code, 2)
        self.assertIn("error:", self.stderr.getvalue())
        self.assertNotIn("Traceback", self.stderr.getvalue())
        if message:
            self.assertIn(message, self.stderr.getvalue())
        if before_model:
            self.yolo.assert_not_called()
            self.caches.assert_not_called()
        self.model.val.assert_not_called()
        self.assertFalse(self.outputs.exists())

    def result(self, name="val_baseline"):
        return json.loads((self.outputs / name / "metrics.json").read_text(encoding="utf-8"))

    def test_help_does_not_load_data_or_model(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--final-test", self.stdout.getvalue())
        self.yolo.assert_not_called()
        self.caches.assert_not_called()
        self.assertFalse(self.outputs.exists())

    def test_invalid_sizes_and_batches_fail_before_data_or_model(self):
        for field in ("--imgsz", "--batch"):
            for value in ("0", "-1", "1.5", "true"):
                with self.subTest(field=field, value=value), patch("src.evaluate.validate", side_effect=AssertionError("Invalid parameter read dataset")):
                    self.rejected([field, value], field.lstrip("-"))

    def test_invalid_names_and_devices_fail_before_model(self):
        for name in ("", "../bad", "a b", "CON", "LPT1"):
            with self.subTest(name=name):
                self.rejected(["--name", name], "name")
        for device in ("", "cuda:0", "-1", "0,1"):
            with self.subTest(device=device):
                self.rejected(["--device", device], "device")

    def test_missing_and_empty_inputs_have_short_errors(self):
        for args in (["--weights", ""], ["--data", ""], ["--weights", str(self.root / "missing.pt")], ["--data", str(self.root / "missing.yaml")]):
            with self.subTest(args=args):
                self.rejected(args)

    def test_final_test_flag_and_split_must_agree(self):
        self.rejected(["--split", "test"], "--final-test")
        self.rejected(["--final-test"], "--split test")

    def test_missing_record_blocks_final_test_but_allows_exploratory_validation(self):
        self.experiment_path.unlink()
        self.rejected(["--split", "test", "--final-test"], "experiment.json")
        self.invoke()
        self.assertIsNone(self.result()["training_record"])
        self.assertEqual(self.result()["split"], "val")

    def test_corrupt_or_non_object_training_records_are_rejected(self):
        for content in (b"[]", b"null", b"{}", b"{broken", b"\xffinvalid"):
            self.experiment_path.write_bytes(content)
            with self.subTest(content=content):
                self.rejected(message="experiment.json")

    def test_record_path_cannot_silently_be_a_directory(self):
        self.experiment_path.unlink()
        self.experiment_path.mkdir()
        self.rejected(message="experiment.json")

    def test_record_fields_are_validated_before_model_loading(self):
        for changes in ({"dataset_fingerprint": None}, {"dataset_fingerprint": "bad"}, {"mode": None}, {"mode": "unknown"}, {"status": None}, {"status": "unknown"}, {"best_weights": None}, {"last_weights": []}):
            write_json(self.experiment_path, dict(self.record, **changes))
            with self.subTest(changes=changes):
                self.rejected(message="experiment.json")

    def test_record_dataset_version_must_match_current_data(self):
        write_json(self.experiment_path, dict(self.record, dataset_fingerprint="0" * 64))
        self.rejected(message="dataset")

    def test_final_test_requires_completed_baseline_and_recorded_checkpoint(self):
        cases = ({"mode": "smoke"}, {"status": "running"}, {"status": "failed"}, {"best_weights": str(self.root / "other.pt"), "last_weights": str(self.root / "other_last.pt")})
        for changes in cases:
            write_json(self.experiment_path, dict(self.record, **changes))
            with self.subTest(changes=changes):
                self.rejected(["--split", "test", "--final-test"])
        missing = dict(self.record)
        missing.pop("best_weights")
        missing.pop("last_weights")
        write_json(self.experiment_path, missing)
        self.rejected(["--split", "test", "--final-test"])

    def test_smoke_or_partial_records_can_still_be_used_for_validation(self):
        for index, changes in enumerate(({"mode": "smoke"}, {"status": "running"}, {"status": "failed"})):
            write_json(self.experiment_path, dict(self.record, **changes))
            name = f"partial_{index}"
            self.invoke(["--name", name])
            self.assertEqual(self.result(name)["training_record"]["mode"], changes.get("mode", "baseline"))
            self.assertEqual(self.result(name)["training_record"]["status"], changes.get("status", "complete"))

    def test_invalid_dataset_errors_include_the_offending_sample(self):
        label = Path(self.data["path"]) / "labels/train/sample_0.txt"
        label.write_text("0 1.1 0.5 0.2 0.2\n", encoding="utf-8")
        self.rejected(message="train/sample_0.png")

    def test_missing_or_stale_review_blocks_model_loading(self):
        self.review_path.unlink()
        self.rejected(message="review")
        write_json(self.review_path, {"dataset_fingerprint": "0" * 64})
        self.rejected(message="review")

    def test_existing_output_is_preserved_before_model_loading(self):
        directory = self.outputs / "val_baseline"
        directory.mkdir(parents=True)
        marker = directory / "keep.txt"
        marker.write_text("existing evaluation", encoding="utf-8")
        with self.assertRaises(SystemExit) as error:
            self.invoke()
        self.assertEqual(error.exception.code, 2)
        self.yolo.assert_not_called()
        self.caches.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing evaluation")
        self.assertEqual(list(directory.iterdir()), [marker])

    def test_output_parent_file_is_rejected_before_model_loading(self):
        parent = self.root / "results"
        parent.write_text("keep existing file", encoding="utf-8")
        self.rejected(message="directory")
        self.assertEqual(parent.read_text(encoding="utf-8"), "keep existing file")

    def test_unavailable_device_has_a_short_error_before_loading_model(self):
        self.device.side_effect = RuntimeError("CUDA device is unavailable")
        self.rejected(message="CUDA")

    def test_model_class_mismatch_is_rejected_before_output_creation(self):
        for names in ({0: "different", 1: "no_mask"}, {0.5: "mask", 1: "no_mask"}, {False: "mask", 1: "no_mask"}, {0: "mask", "0": "mask", 1: "no_mask"}, None):
            self.model.names = names
            with self.subTest(names=names):
                self.rejected(message="class", before_model=False)

    def test_model_loading_and_dependency_errors_leave_no_result_directory(self):
        self.yolo.side_effect = OSError("Cannot load the synthetic fixture")
        self.rejected(message="weights", before_model=False)
        with self.assertRaises(SystemExit) as error:
            self.invoke(backend=False)
        self.assertEqual(error.exception.code, 2)
        self.assertIn("Ultralytics", self.stderr.getvalue())
        self.assertFalse(self.outputs.exists())

    def test_successful_fake_evaluation_records_class_metrics_and_provenance(self):
        self.experiment_path.write_text("\ufeff" + json.dumps(self.record), encoding="utf-8")
        self.invoke()
        result = self.result()
        self.assertEqual(result["weights_sha256"], sha256(self.weights))
        self.assertEqual(result["dataset_fingerprint"], self.fingerprint)
        self.assertEqual(result["training_record"]["path"], str(self.experiment_path))
        self.assertEqual((result["training_record"]["mode"], result["training_record"]["status"]), ("baseline", "complete"))
        self.assertFalse(result["final_test"])
        self.assertEqual(result["parameters"]["batch"], 8)
        self.assertEqual(result["per_class"][0]["precision"], 0.6)
        self.assertEqual(result["per_class"][1]["precision"], 0.8)
        self.assertEqual([row["instances"] for row in result["per_class"]], [1, 1])
        self.model.val.assert_called_once()

    def test_recorded_last_checkpoint_can_be_used_for_final_test(self):
        self.invoke(["--weights", str(self.last), "--split", "test", "--final-test", "--name", "final_last"])
        result = self.result("final_last")
        self.assertEqual(result["weights"], str(self.last))
        self.assertTrue(result["final_test"])
        self.assertEqual(result["split"], "test")

    def test_list_model_names_and_unavailable_class_metrics_remain_supported(self):
        self.model.names = ["mask", "no_mask"]
        self.metrics.box.ap_class_index = [0]
        self.invoke()
        self.assertIsNone(self.result()["per_class"][1]["precision"])
        self.assertIsNone(self.result()["per_class"][1]["mAP50_95"])

    def test_changed_weights_or_data_during_evaluation_cannot_publish_metrics(self):
        def change_weights(**parameters):
            self.weights.write_bytes(b"changed synthetic fixture")
            return self.metrics

        def change_data(**parameters):
            label = Path(self.data["path"]) / "labels/val/sample_1.txt"
            label.write_text("0 0.5 0.5 0.2 0.2\n1 0.75 0.75 0.5 0.5\n", encoding="utf-8")
            return self.metrics

        for index, change in enumerate((change_weights, change_data)):
            name = f"changed_{index}"
            self.model.val.side_effect = change
            with self.subTest(change=index):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(["--name", name])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("changed", self.stderr.getvalue())
                self.assertFalse((self.outputs / name / "metrics.json").exists())
