"""Partial fine-tuning keeps legacy checkpoint guards and requires a verified initializer."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.classification_model import completed_training_record, feature_blocks_to_train
from src.common import sha256, write_json
from src.train_classification import require_preflight, training_plan
from scripts.preflight_classification import validate_runtime_plan


class FeatureScopeTests(unittest.TestCase):
    def test_head_only_default_and_explicit_tail_scope(self):
        self.assertEqual(feature_blocks_to_train({"freeze_backbone": True}), 0)
        for count in (2, 4, 12):
            self.assertEqual(feature_blocks_to_train({"freeze_backbone": False, "train_last_blocks": count}), count)

    def test_invalid_or_contradictory_scope_is_rejected(self):
        for count in (-1, 13, True, 1.5, "2", None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                feature_blocks_to_train({"freeze_backbone": False, "train_last_blocks": count})
        for plan in ({"freeze_backbone": True, "train_last_blocks": 2}, {"freeze_backbone": False}, {}, []):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                feature_blocks_to_train(plan)


class FineTuningRecordTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.weights = self.root / "best.pt"
        self.weights.write_bytes(b"metadata fixture; no model loading")
        self.record = {"task": "classification", "status": "complete", "mode": "baseline",
                       "best_weights_sha256": sha256(self.weights), "checkpoint_reload_verified": True,
                       "classifier_updated": True, "training_strategy": "partial_finetune",
                       "parameters": {"freeze_backbone": False, "train_last_blocks": 2},
                       "frozen_features_unchanged": False, "frozen_feature_prefix_unchanged": True,
                       "trainable_features_updated": True, "feature_buffers_unchanged": True}

    def save(self, record=None):
        write_json(self.root / "experiment.json", record or self.record)

    def test_verified_partial_scope_is_accepted_without_claiming_whole_backbone_frozen(self):
        self.save()
        result = completed_training_record(self.weights)
        self.assertFalse(result["frozen_features_unchanged"])
        self.assertTrue(result["frozen_feature_prefix_unchanged"])

    def test_prefix_buffers_tail_and_classifier_must_all_be_verified(self):
        for key in ("frozen_feature_prefix_unchanged", "feature_buffers_unchanged", "trainable_features_updated",
                    "classifier_updated", "checkpoint_reload_verified"):
            changed = dict(self.record, **{key: False})
            self.save(changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                completed_training_record(self.weights)
        self.save(dict(self.record, frozen_features_unchanged=True))
        with self.assertRaises(ValueError):
            completed_training_record(self.weights)

    def test_changed_weights_or_unknown_strategy_is_rejected(self):
        self.save(dict(self.record, training_strategy="unchecked_full_training"))
        with self.assertRaises(ValueError):
            completed_training_record(self.weights)
        self.save()
        self.weights.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            completed_training_record(self.weights)

    def test_legacy_head_only_still_requires_entire_features_unchanged(self):
        record = copy.deepcopy(self.record)
        record.pop("training_strategy")
        record["parameters"] = {"freeze_backbone": True}
        record["frozen_features_unchanged"] = True
        self.save(record)
        self.assertEqual(completed_training_record(self.weights)["mode"], "baseline")
        record["frozen_features_unchanged"] = False
        self.save(record)
        with self.assertRaises(ValueError):
            completed_training_record(self.weights)

    def test_learning_rates_and_clipping_are_validated_before_training(self):
        config = self.root / "config.yaml"
        config.write_text("fixture: true\n", encoding="utf-8")
        plan = {"freeze_backbone": False, "train_last_blocks": 2, "epochs": 20, "patience": 5,
                "lr": .0001, "weight_decay": .0001, "backbone_lr": .00003, "max_grad_norm": 1,
                "optimizer": "AdamW", "workers": 0, "class_weighting": "inverse_train_frequency",
                "name": "fixture_tail", "project": "results", "preflight": "preflight.json",
                "classes": ["class_0", "class_1"]}
        checked = {"counts": {"train": {"class_0": 3, "class_1": 2}}}
        with patch("src.train_classification.validate_runtime_plan", return_value=(plan, checked)):
            self.assertEqual(training_plan(config)[0]["backbone_lr"], .00003)
        for key, value in (("backbone_lr", 0), ("backbone_lr", float("nan")),
                           ("backbone_lr", .001), ("max_grad_norm", True), ("max_grad_norm", -1)):
            changed = dict(plan, **{key: value})
            with self.subTest(key=key, value=value), patch("src.train_classification.validate_runtime_plan", return_value=(changed, checked)), self.assertRaises(ValueError):
                training_plan(config)

    def test_preflight_binds_initializer_record_and_tail_scope(self):
        config = self.root / "config.yaml"
        config.write_text("fixture: true\n", encoding="utf-8")
        self.save()
        preflight = self.root / "preflight.json"
        plan = {"freeze_backbone": False, "train_last_blocks": 2, "preflight": str(preflight),
                "weights_sha256": "w", "init_checkpoint": str(self.weights), "init_checkpoint_sha256": sha256(self.weights)}
        report = {"status": "ready_before_training", "training_started": False, "gradient_updates": 0,
                  "config_sha256": sha256(config), "initial_weights_sha256": "w",
                  "dataset": {"dataset_fingerprint": "f"}, "train_last_blocks": 2,
                  "initial_classifier_checkpoint_sha256": sha256(self.weights),
                  "initializer_record_sha256": sha256(self.root / "experiment.json")}
        write_json(preflight, report)
        require_preflight(config, plan, {"dataset_fingerprint": "f"})
        for key, value in (("train_last_blocks", 4), ("initial_classifier_checkpoint_sha256", "other"),
                           ("initializer_record_sha256", "other")):
            write_json(preflight, dict(report, **{key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_preflight(config, plan, {"dataset_fingerprint": "f"})

    def runtime_plan(self):
        import yaml
        weights = self.root / "imagenet.pth"
        weights.write_bytes(b"initial weights metadata fixture")
        parent = copy.deepcopy(self.record)
        parent["parameters"]["classes"] = ["class_0", "class_1"]
        parent["dataset_fingerprint"] = "f"
        self.save(parent)
        plan = {"task": "classification", "model": "mobilenet_v3_small", "data": str(self.root / "data"),
                "dataset_fingerprint": "f", "classes": ["class_0", "class_1"], "device": "cpu",
                "imgsz": 224, "batch": 2, "seed": 42, "freeze_backbone": False, "train_last_blocks": 2,
                "weights": str(weights), "weights_sha256": sha256(weights),
                "init_checkpoint": str(self.weights), "init_checkpoint_sha256": sha256(self.weights)}
        config = self.root / "runtime.yaml"
        config.write_text(yaml.safe_dump(plan), encoding="utf-8")
        return config, plan

    def test_runtime_binds_initializer_bytes_and_data(self):
        import yaml
        config, plan = self.runtime_plan()
        checked = {"dataset_fingerprint": "f", "classes": ["class_0", "class_1"]}
        with patch("scripts.preflight_classification.inspect_dataset", return_value=checked):
            self.assertEqual(validate_runtime_plan(config)[0]["init_checkpoint_sha256"], sha256(self.weights))
            for key, value in (("init_checkpoint_sha256", "wrong"), ("init_checkpoint", "")):
                config.write_text(yaml.safe_dump(dict(plan, **{key: value})), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_runtime_plan(config)
            config.write_text(yaml.safe_dump(plan), encoding="utf-8")
            parent = self.record | {"dataset_fingerprint": "other"}
            self.save(parent)
            with self.assertRaises(ValueError):
                validate_runtime_plan(config)

    def test_head_only_cannot_silently_ignore_finetune_options(self):
        import yaml
        config, plan = self.runtime_plan()
        plan.update(freeze_backbone=True, train_last_blocks=0)
        config.write_text(yaml.safe_dump(plan), encoding="utf-8")
        with patch("scripts.preflight_classification.inspect_dataset", return_value={"dataset_fingerprint": "f", "classes": plan["classes"]}), self.assertRaisesRegex(ValueError, "require train_last_blocks"):
            validate_runtime_plan(config)


if __name__ == "__main__":
    unittest.main()
