"""Prediction preflight and streaming records, with no real model or video decoder."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.common import sha256, write_json
from src.predict import main as predict_main


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def boxes(coordinates=None, categories=None, confidence=None):
    return types.SimpleNamespace(xyxy=FakeTensor([[1, 2, 6, 7]] if coordinates is None else coordinates), cls=FakeTensor([0.0] if categories is None else categories), conf=FakeTensor([0.9] if confidence is None else confidence))


class PredictionTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        inputs = self.root / "输入图片"
        inputs.mkdir()
        self.source = inputs / "sample.PNG"
        Image.new("RGB", (10, 12), (30, 40, 50)).save(self.source)
        self.original_image = self.source.read_bytes()
        self.weights = self.root / "model.pt"
        self.weights.write_bytes(b"synthetic fixture, not model weights")
        self.outputs = self.root / "results/predictions"
        self.model = types.SimpleNamespace(task="detect", names={0: "mask", 1: "no_mask"}, predict=Mock(side_effect=lambda **kwargs: iter([self.frame()])))
        self.yolo = Mock(return_value=self.model)
        self.backend = types.ModuleType("ultralytics")
        self.backend.YOLO = self.yolo
        self.caches, self.device = Mock(), Mock(return_value="cpu")

    def frame(self, path=..., **changes):
        result_path = self.source if path is ... else path
        values = {"path": str(result_path) if result_path is not None else None, "orig_shape": (12, 10), "names": self.model.names, "boxes": boxes()}
        values.update(changes)
        return types.SimpleNamespace(**values)

    def invoke(self, extra=(), backend=True):
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        argv = ["predict", "--source", str(self.source), "--weights", str(self.weights), "--device", "cpu", *extra]
        with patch("sys.argv", argv), patch("src.common.ROOT", self.root), patch.dict("sys.modules", {"ultralytics": self.backend if backend else None}), patch("src.predict.local_caches", self.caches), patch("src.predict.select_device", self.device), contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            predict_main()

    def rejected(self, extra=(), message=None, before_model=True):
        with self.assertRaises(SystemExit) as error:
            self.invoke(extra)
        self.assertEqual(error.exception.code, 2)
        self.assertNotIn("Traceback", self.stderr.getvalue())
        if message:
            self.assertIn(message, self.stderr.getvalue())
        if before_model:
            self.yolo.assert_not_called()
            self.caches.assert_not_called()
        self.model.predict.assert_not_called()
        self.assertFalse(self.outputs.exists())

    def record(self, name="baseline_examples"):
        return json.loads((self.outputs / name / "run.json").read_text(encoding="utf-8"))

    def failed(self, name, extra=(), message=None):
        with self.assertRaises(SystemExit) as error:
            self.invoke(["--name", name, *extra])
        self.assertEqual(error.exception.code, 2)
        record = self.record(name)
        self.assertEqual(record["status"], "failed")
        if message:
            self.assertIn(message, record["error"])
        self.assertFalse((self.outputs / name / "predictions.jsonl").exists())
        self.assertTrue((self.outputs / name / "predictions.jsonl.part").is_file())
        return record

    def test_help_does_not_load_model_or_create_outputs(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--source", self.stdout.getvalue())
        self.yolo.assert_not_called()
        self.caches.assert_not_called()
        self.assertFalse(self.outputs.exists())

    def test_confidence_and_size_parameters_are_checked_before_model(self):
        for value in ("-0.1", "1.1", "nan", "inf", "-inf"):
            with self.subTest(conf=value):
                self.rejected(["--conf", value], "conf")
        for value in ("0", "-1", "0.5", "true"):
            with self.subTest(imgsz=value):
                self.rejected(["--imgsz", value], "imgsz")

    def test_invalid_names_and_devices_are_checked_before_model(self):
        for value in ("", "../escape", "a b", "CON", "LPT1"):
            with self.subTest(name=value):
                self.rejected(["--name", value], "name")
        for value in ("", "cuda:0", "0,1", "-1"):
            with self.subTest(device=value):
                self.rejected(["--device", value], "device")

    def test_missing_or_empty_paths_have_short_errors(self):
        for extra in (["--source", ""], ["--weights", ""], ["--source", str(self.root / "missing.png")], ["--weights", str(self.root / "missing.pt")]):
            with self.subTest(extra=extra):
                self.rejected(extra)

    def test_unsupported_files_and_directories_without_direct_media_are_rejected(self):
        text = self.root / "sources.txt"
        text.write_text("not a local media file", encoding="utf-8")
        empty = self.root / "empty"
        empty.mkdir()
        nested = self.root / "nested"
        (nested / "child").mkdir(parents=True)
        (nested / "child/sample.png").write_bytes(self.original_image)
        for path in (text, empty, nested):
            with self.subTest(path=path):
                self.rejected(["--source", str(path)], "source")

    def test_empty_media_files_are_rejected_before_loading_model(self):
        path = self.root / "empty.mp4"
        path.touch()
        self.rejected(["--source", str(path)], "empty")

    def test_existing_results_are_not_overwritten_or_loaded_into_model(self):
        directory = self.outputs / "baseline_examples"
        directory.mkdir(parents=True)
        marker = directory / "keep.txt"
        marker.write_text("existing result", encoding="utf-8")
        with self.assertRaises(SystemExit) as error:
            self.invoke()
        self.assertEqual(error.exception.code, 2)
        self.yolo.assert_not_called()
        self.caches.assert_not_called()
        self.assertEqual(list(directory.iterdir()), [marker])
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing result")

    def test_output_parent_file_is_rejected_before_loading_model(self):
        parent = self.root / "results"
        parent.write_text("keep existing file", encoding="utf-8")
        self.rejected(message="directory")
        self.assertEqual(parent.read_text(encoding="utf-8"), "keep existing file")

    def test_device_and_dependency_errors_do_not_create_result_directory(self):
        self.device.side_effect = RuntimeError("CUDA is unavailable")
        self.rejected(message="CUDA")
        self.device.side_effect = None
        with self.assertRaises(SystemExit) as error:
            self.invoke(backend=False)
        self.assertEqual(error.exception.code, 2)
        self.assertIn("Ultralytics", self.stderr.getvalue())
        self.assertFalse(self.outputs.exists())

    def test_model_loading_error_precedes_output_creation(self):
        self.yolo.side_effect = OSError("Cannot load synthetic fixture")
        self.rejected(message="weights", before_model=False)

    def test_non_detection_models_are_rejected(self):
        self.model.task = "classify"
        self.rejected(message="detection", before_model=False)

    def test_invalid_model_class_names_precede_output_creation(self):
        for names in ({0.5: "mask"}, {False: "mask"}, {0: "mask", "0": "other"}, ["mask", "mask"], [], None):
            self.model.names = names
            with self.subTest(names=names):
                self.rejected(message="class", before_model=False)

    def test_successful_jsonl_and_run_record_bind_the_actual_inputs(self):
        self.model.names = ["正确佩戴", "未佩戴"]
        self.invoke(["--conf", "0"])
        directory = self.outputs / "baseline_examples"
        row = json.loads((directory / "predictions.jsonl").read_text(encoding="utf-8"))
        record = self.record()
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["weights_sha256"], sha256(self.weights))
        self.assertEqual(record["source_files"], [{"path": str(self.source), "sha256": sha256(self.source)}])
        self.assertEqual(record["images_or_frames"], 1)
        self.assertEqual(row["detections"][0]["name"], "正确佩戴")
        self.assertEqual(row["detections"][0]["xyxy_pixels"], [1.0, 2.0, 6.0, 7.0])
        self.assertEqual(row["image_size"], {"width": 10, "height": 12})
        self.assertEqual((row["frame_index"], row["source_frame_index"]), (0, 0))
        self.assertFalse((directory / "predictions.jsonl.part").exists())
        self.assertTrue(self.model.predict.call_args.kwargs["stream"])

    def test_directory_results_have_global_and_per_source_indices(self):
        second = self.source.with_name("second.jpg")
        second.write_bytes(self.original_image)
        self.model.predict.side_effect = lambda **kwargs: iter([self.frame(), self.frame(second, boxes=None)])
        self.invoke(["--source", str(self.source.parent)])
        rows = [json.loads(line) for line in (self.outputs / "baseline_examples/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["frame_index"] for row in rows], [0, 1])
        self.assertEqual([row["source_frame_index"] for row in rows], [0, 0])
        self.assertEqual(rows[1]["detections"], [])
        self.assertEqual(self.record()["images_or_frames"], 2)

    def test_video_stream_is_written_incrementally_and_closed(self):
        video = self.root / "sample.MP4"
        video.write_bytes(b"synthetic video fixture; decoder not used")
        closed = []

        def stream(**kwargs):
            try:
                yield self.frame(video)
                part = self.outputs / "baseline_examples/predictions.jsonl.part"
                self.assertEqual(len(part.read_text(encoding="utf-8").splitlines()), 1)
                yield self.frame(video)
                yield self.frame(video, boxes=None)
            finally:
                closed.append(True)

        self.model.predict.side_effect = stream
        self.invoke(["--source", str(video), "--conf", "1"])
        rows = [json.loads(line) for line in (self.outputs / "baseline_examples/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["source_frame_index"] for row in rows], [0, 1, 2])
        self.assertEqual(self.record()["images_or_frames"], 3)
        self.assertEqual(closed, [True])

    def test_different_detection_array_lengths_cannot_be_silently_truncated(self):
        self.model.predict.side_effect = lambda **kwargs: iter([self.frame(boxes=boxes(categories=[0.0, 1.0]))])
        record = self.failed("unequal", message="length")
        self.assertEqual(record["images_or_frames"], 0)

    def test_invalid_box_coordinates_cannot_be_published(self):
        invalid = ([1, 2, float("nan"), 7], [1, 2, float("inf"), 7], [-1, 2, 6, 7], [6, 2, 1, 7], [1, 2, 1, 7], [1, 2, 11, 7], [1, 2, 6], [True, 2, 6, 7])
        for index, coordinates in enumerate(invalid):
            with self.subTest(coordinates=coordinates):
                self.model.predict.side_effect = lambda coordinates=coordinates, **kwargs: iter([self.frame(boxes=boxes(coordinates=[coordinates]))])
                self.failed(f"box_{index}")

    def test_invalid_categories_and_confidence_cannot_be_published(self):
        invalid = ((0.5, 0.9), (-1, 0.9), (2, 0.9), (True, 0.9), (float("nan"), 0.9), (0, 1.1), (0, -0.1), (0, float("inf")), (0, float("nan")), (0, False))
        for index, (category, confidence) in enumerate(invalid):
            with self.subTest(category=category, confidence=confidence):
                self.model.predict.side_effect = lambda category=category, confidence=confidence, **kwargs: iter([self.frame(boxes=boxes(categories=[category], confidence=[confidence]))])
                self.failed(f"values_{index}")

    def test_invalid_frame_metadata_cannot_be_published(self):
        invalid = ({"path": str(self.root / "unexpected.png")}, {"path": None}, {"orig_shape": (0, 10)}, {"orig_shape": (True, 10)}, {"orig_shape": None}, {"names": {0: "different", 1: "no_mask"}})
        for index, changes in enumerate(invalid):
            with self.subTest(changes=changes):
                self.model.predict.side_effect = lambda changes=changes, **kwargs: iter([self.frame(**changes)])
                self.failed(f"metadata_{index}")

    def test_zero_output_and_missing_input_results_are_not_complete(self):
        self.model.predict.side_effect = lambda **kwargs: iter([])
        record = self.failed("no_frames")
        self.assertEqual(record["images_or_frames"], 0)
        second = self.source.with_name("skipped.png")
        second.write_bytes(self.original_image)
        self.model.predict.side_effect = lambda **kwargs: iter([self.frame()])
        self.failed("skipped", ["--source", str(self.source.parent)], message="missing")

    def test_duplicate_image_results_are_not_complete(self):
        self.model.predict.side_effect = lambda **kwargs: iter([self.frame(), self.frame()])
        self.failed("duplicate", message="Duplicate")

    def test_backend_error_preserves_partial_jsonl_and_closes_stream(self):
        closed = []

        def stream(**kwargs):
            try:
                yield self.frame()
                raise RuntimeError("Synthetic stream failure")
            finally:
                closed.append(True)

        self.model.predict.side_effect = stream
        record = self.failed("interrupted_stream", message="Synthetic stream failure")
        self.assertEqual(record["images_or_frames"], 1)
        part = self.outputs / "interrupted_stream/predictions.jsonl.part"
        self.assertEqual(len(part.read_text(encoding="utf-8").splitlines()), 1)
        self.assertEqual(closed, [True])

    def test_keyboard_interrupt_records_failure_and_preserves_partial_output(self):
        def stream(**kwargs):
            yield self.frame()
            raise KeyboardInterrupt()

        self.model.predict.side_effect = stream
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(["--name", "keyboard"])
        record = self.record("keyboard")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "KeyboardInterrupt")
        self.assertEqual(record["images_or_frames"], 1)
        self.assertFalse((self.outputs / "keyboard/predictions.jsonl").exists())

    def test_changed_weights_sources_or_membership_do_not_publish_results(self):
        for index in range(3):
            self.source.write_bytes(self.original_image)

            def stream(index=index, **kwargs):
                yield self.frame()
                if index == 0:
                    self.weights.write_bytes(b"changed synthetic fixture")
                elif index == 1:
                    self.source.write_bytes(b"changed source content")
                else:
                    self.source.with_name("added.png").write_bytes(self.original_image)

            self.model.predict.side_effect = stream
            with self.subTest(change=index):
                extra = ["--source", str(self.source.parent)] if index == 2 else []
                self.failed(f"changed_{index}", extra, message="changed")

    def test_publication_and_completion_record_errors_are_not_complete(self):
        original_replace = Path.replace

        def failing_replace(path, target):
            if path.name == "predictions.jsonl.part":
                raise PermissionError("Synthetic publication failure")
            return original_replace(path, target)

        with patch.object(Path, "replace", failing_replace):
            self.failed("publish_error", message="publication failure")

        def failing_completion(path, record):
            if record.get("status") == "complete":
                raise OSError("Synthetic completion record failure")
            return write_json(path, record)

        with patch("src.predict.write_json", side_effect=failing_completion):
            self.failed("record_error", message="completion record failure")

    def test_failed_completion_and_blocked_rollback_report_unconfirmed_file(self):
        original_replace = Path.replace

        def blocked_rollback(path, target):
            if path.name == "predictions.jsonl":
                raise PermissionError("Synthetic rollback blocked")
            return original_replace(path, target)

        def failing_completion(path, record):
            if record.get("status") == "complete":
                raise OSError("Synthetic completion record failure")
            return write_json(path, record)

        with patch.object(Path, "replace", blocked_rollback), patch("src.predict.write_json", side_effect=failing_completion), self.assertRaises(SystemExit) as error:
            self.invoke(["--name", "blocked_rollback"])
        self.assertEqual(error.exception.code, 2)
        record = self.record("blocked_rollback")
        self.assertEqual(record["status"], "failed")
        self.assertIsNone(record["predictions_file"])
        self.assertIsNone(record["partial_predictions_file"])
        self.assertIn("rollback blocked", record["publication_rollback_error"])
        remaining = self.outputs / "blocked_rollback/predictions.jsonl"
        self.assertEqual(record["unconfirmed_predictions_file"], str(remaining))
        self.assertTrue(remaining.is_file())
