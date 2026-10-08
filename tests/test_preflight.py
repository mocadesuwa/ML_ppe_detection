"""A readiness report must not be produced for stale or incomplete inputs."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.preflight_training import run
from src.common import dataset_config, dataset_fingerprint, write_json


class PreflightGateTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        import yaml
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "data"
        for i, split in enumerate(("train", "val", "test")):
            images, labels = self.base / "images" / split, self.base / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (8, 8), (i * 50, 20, 30)).save(images / f"{i}.png")
            (labels / f"{i}.txt").write_text("0 .25 .25 .5 .5\n1 .75 .75 .5 .5\n", encoding="utf-8")
        source = self.root / "data.yaml"
        source.write_text(yaml.safe_dump({"path": str(self.base), "train": "images/train", "val": "images/val", "test": "images/test", "names": ["mask", "no_mask"]}), encoding="utf-8")
        self.data = dataset_config(source)
        write_json(self.base / "quality_review.json", {"dataset_fingerprint": dataset_fingerprint(self.data)})
        self.weights = self.root / "initial.pt"
        self.weights.write_bytes(b"synthetic preflight gate fixture")
        self.output = self.root / "evidence"
        self.plan = {"model": str(self.weights), "device": "auto"}
        self.args = argparse.Namespace(output=str(self.output))

    def reject(self, error, message):
        with patch("scripts.preflight_training.build_plan", return_value=(self.plan, self.data)), \
             patch("scripts.preflight_training.inspect_environment") as environment, \
             self.assertRaisesRegex(error, message):
            run(self.args)
        environment.assert_not_called()
        self.assertFalse((self.output / "report.json").exists())

    def test_missing_labels_do_not_produce_readiness(self):
        (self.base / "labels/train/0.txt").unlink()
        self.reject(ValueError, "dataset errors")

    def test_changed_valid_labels_invalidate_the_review(self):
        (self.base / "labels/train/0.txt").write_text("0 .3 .3 .4 .4\n1 .7 .7 .4 .4\n", encoding="utf-8")
        self.reject(RuntimeError, "changed after review")

    def test_missing_local_weights_do_not_trigger_a_download(self):
        self.weights.unlink()
        self.reject(FileNotFoundError, "local initial weights")

    def test_existing_report_directory_is_preserved(self):
        self.output.mkdir()
        kept = self.output / "keep.txt"
        kept.write_text("previous evidence", encoding="utf-8")
        self.reject(FileExistsError, "output exists")
        self.assertEqual(kept.read_text(encoding="utf-8"), "previous evidence")

    def test_evidence_cannot_be_written_inside_frozen_data(self):
        self.output = self.base / "new_evidence"
        self.args.output = str(self.output)
        self.reject(ValueError, "outside the frozen dataset")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
