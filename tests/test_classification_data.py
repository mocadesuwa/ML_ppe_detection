"""Meaningful integrity and leakage checks using disposable neutral fixtures."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.check_head_crops import inspect_crops
from src.classification_data import export_dataset, inspect_dataset
from src.common import write_json
from src.prepare_head_crops import build_plan, prepare
from src.split_head_crops import build_split
from scripts.preflight_classification import run as run_preflight, validate_runtime_plan
from src.train_classification import training_plan, start as start_training


class ClassificationDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "source"
        source.mkdir()
        items = []
        for group in range(6):
            for label in range(2):
                identifier = f"sample_{group}_{label}"
                path = source / f"{identifier}.png"
                Image.new("RGB", (64, 64), (group * 30, label * 100, group + label)).save(path)
                items.append({"id": identifier, "source": str(path), "box_xyxy": [0, 0, 64, 64],
                              "group": f"group_{group}", "label": f"class_{label}"})
        self.plan_file = self.root / "heads.json"
        write_json(self.plan_file, {"format_version": 1, "items": items})
        self.crops = self.root / "crops"
        prepare(build_plan(self.plan_file, self.crops))
        checked = inspect_crops(self.crops)
        self.split_file = self.root / "splits.json"
        write_json(self.split_file, build_split(self.crops, ["class_0", "class_1"], [.5, .25, .25], 42))
        self.review_file = self.root / "review.json"
        write_json(self.review_file, {"format_version": 1, "reviewer": "synthetic test fixture",
                                     "visual_review_completed": True, "grouping_review_completed": True,
                                     "crop_fingerprint": checked["crop_fingerprint"],
                                     "accepted_ids": [item["id"] for item in checked["crops"]]})
        self.output = self.root / "classification"

    def export(self, **kwargs):
        return export_dataset(self.crops, self.split_file, self.review_file, self.output, **kwargs)

    def change(self, path, mutation):
        value = json.loads(path.read_text(encoding="utf-8"))
        mutation(value)
        write_json(path, value)

    def test_roundtrip_and_dry_run(self):
        planned = self.export(dry_run=True)
        self.assertFalse(self.output.exists())
        exported = self.export()
        self.assertEqual(planned["dataset_fingerprint"], exported["dataset_fingerprint"])
        checked = inspect_dataset(self.output)
        self.assertEqual(checked["image_count"], 12)
        self.assertEqual(checked["source_group_cross_split_count"], 0)
        self.assertTrue(checked["data_ready"])

    def test_incomplete_or_stale_review_rejected(self):
        for field, value in (("visual_review_completed", False), ("grouping_review_completed", False),
                             ("crop_fingerprint", "stale"), ("accepted_ids", []), ("reviewer", "")):
            with self.subTest(field=field):
                original = self.review_file.read_bytes()
                self.change(self.review_file, lambda review: review.update({field: value}))
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse(self.output.exists())
                self.review_file.write_bytes(original)

    def test_source_group_leakage_rejected(self):
        def mutate(split):
            train, val = split["splits"]["train"], split["splits"]["val"]
            train[0], val[0] = val[0], train[0]
        self.change(self.split_file, mutate)
        with self.assertRaisesRegex(ValueError, "cross splits"):
            self.export()

    def test_changed_labels_and_unsafe_paths_rejected(self):
        for key, value in (("label", "class_9"), ("output_file", "../outside.png")):
            with self.subTest(field=key):
                original = self.split_file.read_bytes()
                self.change(self.split_file, lambda split: split["splits"]["train"][0].update({key: value}))
                with self.assertRaises(ValueError):
                    self.export()
                self.split_file.write_bytes(original)

    def test_tampered_export_and_extra_files_rejected(self):
        exported = self.export()
        image = self.output / exported["splits"]["train"][0]["file"]
        original = image.read_bytes()
        image.write_bytes(b"wrong pixels")
        with self.assertRaises(ValueError):
            inspect_dataset(self.output)
        image.write_bytes(original)
        (self.output / "extra.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "extra files"):
            inspect_dataset(self.output)

    def test_overwrite_and_nested_output_rejected(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.export()
        with self.assertRaises(ValueError):
            export_dataset(self.crops, self.split_file, self.review_file, self.crops / "nested")

    def test_manifest_and_review_tampering_rejected(self):
        self.export()
        self.change(self.output / "manifest.json", lambda manifest: manifest.update(classes=["different"]))
        with self.assertRaises(ValueError):
            inspect_dataset(self.output)

    def test_changed_attached_audit_invalidates_export(self):
        from src.common import sha256
        audit = self.root / "audit.json"
        audit.write_text("{}", encoding="utf-8")
        self.change(self.review_file, lambda review: review.update(audit_file=str(audit), audit_sha256=sha256(audit)))
        self.export()
        audit.write_text('{"changed":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "audit attachment changed"):
            inspect_dataset(self.output)

    def test_training_source_class_confound_is_rejected(self):
        split = json.loads(self.split_file.read_text(encoding="utf-8"))
        domains = {row["id"]: "source_a" for rows in split["splits"].values() for row in rows}
        for row in split["splits"]["val"]:
            domains[row["id"]] = "held_out_only_source"
        self.change(self.review_file, lambda review: review.update(source_domains=domains, require_training_domain_coverage=True))
        with self.assertRaisesRegex(ValueError, "every class in training"):
            self.export()

    def runtime_config(self):
        import yaml
        from src.common import sha256
        manifest = self.export()
        weights = self.root / "fixture_weights.pth"
        weights.write_bytes(b"not a model; validation-only fixture")
        plan = {"task": "classification", "model": "mobilenet_v3_small", "data": str(self.output),
                "dataset_fingerprint": manifest["dataset_fingerprint"], "classes": ["class_0", "class_1"],
                "weights": str(weights), "weights_sha256": sha256(weights), "device": "cpu",
                "imgsz": 224, "batch": 2, "seed": 42, "freeze_backbone": True}
        path = self.root / "runtime.yaml"
        path.write_text(yaml.safe_dump(plan), encoding="utf-8")
        return path, plan

    def test_runtime_gate_checks_data_and_weight_hash_before_model_loading(self):
        import yaml
        path, plan = self.runtime_config()
        validated, checked = validate_runtime_plan(path)
        self.assertEqual(checked["image_count"], 12)
        Path(validated["weights"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "weights changed"):
            validate_runtime_plan(path)
        plan["dataset_fingerprint"] = "wrong"
        path.write_text(yaml.safe_dump(plan), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_runtime_plan(path)

    def test_runtime_output_is_outside_data_and_cannot_overwrite(self):
        path, _ = self.runtime_config()
        with self.assertRaisesRegex(ValueError, "outside frozen data"):
            run_preflight(path, self.output / "evidence")
        with self.assertRaises(FileExistsError):
            run_preflight(path, self.output)

    def test_training_requires_matching_preflight_before_creating_a_run(self):
        import yaml
        path, plan = self.runtime_config()
        plan.update(epochs=30, patience=8, lr=.0003, weight_decay=.0001, optimizer="AdamW", workers=0,
                    class_weighting="inverse_train_frequency", name="fixture_run",
                    project=str(self.root / "training"), preflight=str(self.root / "missing_preflight.json"))
        path.write_text(yaml.safe_dump(plan), encoding="utf-8")
        validated, checked = training_plan(path)
        with self.assertRaises(FileNotFoundError):
            start_training(path, validated, checked)
        self.assertFalse((self.root / "training").exists())
        plan["lr"] = float("nan")
        path.write_text(yaml.safe_dump(plan), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid lr"):
            training_plan(path)

    def test_uniform_class_weights_and_source_sampling_are_explicit(self):
        import yaml
        path, plan = self.runtime_config()
        plan.update(epochs=30, patience=8, lr=.0003, weight_decay=.0001, optimizer="AdamW", workers=0,
                    class_weighting="none", sampling="source_group_balanced", name="fixture_run",
                    project=str(self.root / "training"), preflight=str(self.root / "preflight.json"))
        path.write_text(yaml.safe_dump(plan), encoding="utf-8")
        validated, _ = training_plan(path)
        self.assertEqual(validated["class_weights"], [1.0, 1.0])
        self.assertEqual(validated["sampling"], "source_group_balanced")
        for field, value in (("sampling", "unknown"), ("class_weighting", "from_test")):
            changed = dict(plan, **{field: value})
            path.write_text(yaml.safe_dump(changed), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(ValueError):
                training_plan(path)

    def test_joint_sampler_requires_unweighted_loss(self):
        import yaml
        path, plan = self.runtime_config()
        plan.update(epochs=30, patience=8, lr=.0003, weight_decay=.0001, optimizer="AdamW", workers=0,
                    class_weighting="none", sampling="class_source_group_balanced", name="fixture_joint",
                    project=str(self.root / "training"), preflight=str(self.root / "preflight.json"))
        path.write_text(yaml.safe_dump(plan), encoding="utf-8")
        self.assertEqual(training_plan(path)[0]["sampling"], "class_source_group_balanced")
        plan["class_weighting"] = "inverse_train_frequency"
        path.write_text(yaml.safe_dump(plan), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "double balancing"):
            training_plan(path)


if __name__ == "__main__":
    unittest.main()
