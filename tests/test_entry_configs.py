"""Configuration preflight tests; no model, real dataset or GPU is used."""
from __future__ import annotations

import argparse
import builtins
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common import ROOT, dataset_config, load_yaml, run_directory, select_device
from src.train import build_plan, main as train_main


class EntryConfigTests(unittest.TestCase):
    def setUp(self):
        import yaml

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "配置与路径 tests"
        self.root.mkdir()
        self.data_path = self.root / "dataset.yaml"
        self.data_config = {"path": "../prepared", "train": "images/train", "val": "images/val", "test": "images/test", "names": ["mask"]}
        self.data_path.write_text(yaml.safe_dump(self.data_config), encoding="utf-8")
        self.plan_path = self.root / "baseline.yaml"
        self.config = {"data": str(self.data_path), "model": str(self.root / "weights" / "initial.pt"), "project": str(self.root / "results"), "name": "baseline_01", "epochs": 50, "batch": 8, "imgsz": 640, "device": "auto", "patience": 15, "seed": 42, "workers": 0, "fraction": 1.0, "amp": True, "deterministic": True, "plots": True, "save": True, "cache": False}

    def write_plan(self, changes=None):
        import yaml

        config = dict(self.config, **(changes or {}))
        self.plan_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return config

    def plan(self, changes=None, **overrides):
        self.write_plan(changes)
        args = argparse.Namespace(config=str(self.plan_path), smoke=False, name=None, epochs=None, batch=None, device=None)
        vars(args).update(overrides)
        return build_plan(args)

    def test_config_duplicate_keys_are_not_silently_overwritten(self):
        for content in ("epochs: 50\nepochs: 1\n", "names:\n  0: mask\n  0: other\n", "names:\n  0: mask\n  false: other\n", "[a, b]: value\n"):
            self.plan_path.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "baseline.yaml"):
                load_yaml(self.plan_path)

    def test_config_invalid_yaml_has_file_context(self):
        for content in ("epochs: [broken", "[]", "null", "", "epochs: 1\n---\nepochs: 2\n"):
            self.plan_path.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "baseline.yaml"):
                load_yaml(self.plan_path)

    def test_config_unreadable_files_have_file_context(self):
        for path in (self.plan_path, self.root):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, str(path.name)):
                load_yaml(path)
        self.plan_path.write_bytes(b"\xff\xfeinvalid")
        with self.assertRaisesRegex(ValueError, "baseline.yaml"):
            load_yaml(self.plan_path)

    def test_config_yaml_aliases_and_merge_overrides_remain_supported(self):
        self.plan_path.write_text("\ufeffdefaults: &defaults\n  epochs: 50\n  batch: 8\nrun:\n  <<: *defaults\n  epochs: 3\ncopy: *defaults\nequals:\n  =: value\n", encoding="utf-8")
        loaded = load_yaml(self.plan_path)
        self.assertEqual(loaded["run"], {"epochs": 3, "batch": 8})
        self.assertEqual(loaded["copy"], {"epochs": 50, "batch": 8})
        self.assertEqual(loaded["equals"], {"=": "value"})

    def test_config_dataset_paths_reject_empty_or_non_string_values(self):
        import yaml

        for field in ("path", "train", "val", "test"):
            for value in ("", " \t", None, True, 0, [], {}):
                self.data_path.write_text(yaml.safe_dump(dict(self.data_config, **{field: value})), encoding="utf-8")
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    dataset_config(self.data_path)

    def test_config_dataset_relative_paths_are_independent_of_working_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            config = dataset_config(self.data_path)
        finally:
            os.chdir(previous)
        expected = (self.root / "../prepared").resolve()
        self.assertEqual(config["path"], str(expected))
        self.assertEqual(config["train"], str(expected / "images/train"))
        self.assertEqual(config["names"], {0: "mask"})

    def test_plan_missing_required_fields_report_the_field(self):
        import yaml

        for field in ("data", "model", "project", "name", "epochs", "batch", "imgsz", "device"):
            config = dict(self.config)
            del config[field]
            self.plan_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            args = argparse.Namespace(config=str(self.plan_path), smoke=False, name=None, epochs=None, batch=None, device=None)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                build_plan(args)

    def test_plan_integer_parameters_reject_fractions_booleans_and_bad_ranges(self):
        for field in ("epochs", "batch", "imgsz", "patience", "workers", "seed"):
            invalid = (-1, 0.5, True, False, "3", None, float("inf"), float("nan"))
            if field in ("epochs", "batch", "imgsz"):
                invalid += (0,)
            for value in invalid:
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    self.plan({field: value})
        config, _ = self.plan({"patience": 0, "workers": 0, "seed": 0})
        self.assertEqual((config["patience"], config["workers"], config["seed"]), (0, 0, 0))

    def test_plan_boolean_switches_and_cache_have_explicit_types(self):
        for field in ("amp", "deterministic", "plots", "save"):
            for value in ("false", 0, 1, None):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    self.plan({field: value})
        for value in ("false", "memory", 0, 1, None):
            with self.subTest(cache=value), self.assertRaisesRegex(ValueError, "cache"):
                self.plan({"cache": value})
        for value in (False, True, "ram", "disk"):
            config, _ = self.plan({"cache": value})
            self.assertEqual(config["cache"], value)

    def test_plan_fraction_is_finite_and_in_range(self):
        for value in (0, -0.1, 1.1, 10 ** 400, True, "0.1", None, float("inf"), float("nan")):
            with self.subTest(fraction=value), self.assertRaisesRegex(ValueError, "fraction"):
                self.plan({"fraction": value})
        self.assertEqual(self.plan({"fraction": 1})[0]["fraction"], 1)

    def test_plan_device_syntax_is_checked_without_gpu_dependencies(self):
        for value in ("auto", "cpu", "0", 0, "2", 2):
            with self.subTest(device=value):
                self.assertEqual(self.plan({"device": value})[0]["device"], str(value))
        for value in ("", " ", "cuda:0", "0,1", "-1", -1, True, 0.5, None, []):
            with self.subTest(device=value), self.assertRaisesRegex(ValueError, "device"):
                self.plan({"device": value})

    def test_plan_names_are_checked_before_creating_a_run(self):
        for value in ("", "../escape", "a/b", "a\\b", "a b", "CON", "nul", "COM1", "LPT9", None, True):
            with self.subTest(name=value), self.assertRaisesRegex(ValueError, "name"):
                self.plan({"name": value})
        self.assertEqual(self.plan({"name": "baseline-02_check"})[0]["name"], "baseline-02_check")
        self.assertFalse((self.root / "results").exists())

    def test_plan_path_fields_are_checked_before_resolution(self):
        for field in ("data", "model", "project"):
            for value in ("", " \t", None, True, 0, []):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    self.plan({field: value})

    def test_plan_project_cannot_be_an_existing_file(self):
        with self.assertRaisesRegex(ValueError, "project"):
            self.plan({"project": str(self.data_path)})

    def test_plan_smoke_and_cli_overrides_preserve_priority(self):
        config, _ = self.plan(smoke=True, name="custom_check", epochs=2, batch=4, device="cpu")
        self.assertEqual({key: config[key] for key in ("name", "epochs", "batch", "device", "patience", "fraction")}, {"name": "custom_check", "epochs": 2, "batch": 4, "device": "cpu", "patience": 3, "fraction": 0.1})
        self.assertEqual(load_yaml(self.plan_path), self.config)
        with self.assertRaisesRegex(ValueError, "epochs"):
            self.plan(smoke=True, epochs=0)

    def test_plan_default_baseline_keeps_existing_paths_and_parameters(self):
        args = argparse.Namespace(config="configs/baseline.yaml", smoke=False, name=None, epochs=None, batch=None, device=None)
        config, data = build_plan(args)
        self.assertEqual((config["epochs"], config["batch"], config["imgsz"]), (50, 8, 640))
        self.assertEqual(config["model"], str(ROOT / "weights/yolo11n.pt"))
        self.assertEqual(config["project"], str(ROOT / "results"))
        self.assertEqual(data["path"], str(ROOT / "dataset"))
        self.assertEqual(data["names"], {0: "mask", 1: "no_mask", 2: "mask_incorrect"})

    def test_train_cli_dry_run_prints_plans_without_model_imports_or_writes(self):
        original_import = builtins.__import__

        def without_model(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "ultralytics", "PIL"):
                raise AssertionError(f"Dry run imported {name}")
            return original_import(name, *args, **kwargs)

        self.write_plan()
        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        for smoke in (False, True):
            output = io.StringIO()
            argv = ["train", "--config", str(self.plan_path), "--dry-run", "--device", "cpu"]
            if smoke:
                argv.append("--smoke")
            with patch("sys.argv", argv), patch("builtins.__import__", side_effect=without_model), patch("src.train.validate", side_effect=AssertionError("Dry run read dataset contents")), patch("src.train.local_caches", side_effect=AssertionError("Dry run created caches")), contextlib.redirect_stdout(output):
                train_main()
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["mode"], "smoke" if smoke else "baseline")
            self.assertEqual(plan["train"]["epochs"], 3 if smoke else 50)
            self.assertEqual(plan["train"]["device"], "cpu")
        self.assertEqual(before, sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*")))

    def test_train_cli_invalid_config_and_overrides_exit_with_short_errors(self):
        self.write_plan()
        cases = (["--config", str(self.root / "missing.yaml")], ["--config", str(self.plan_path), "--epochs", "0"], ["--config", str(self.plan_path), "--batch", "0"], ["--config", str(self.plan_path), "--name", "../bad"], ["--config", str(self.plan_path), "--device", "cuda:0"])
        for extra in cases:
            stderr = io.StringIO()
            with self.subTest(extra=extra), patch("sys.argv", ["train", *extra]), patch("src.train.validate", side_effect=AssertionError("Invalid plan reached dataset validation")), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                train_main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("error:", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
        self.assertFalse((self.root / "results").exists())

    def test_train_cli_help_does_not_read_config(self):
        stdout = io.StringIO()
        with patch("sys.argv", ["train", "--help"]), patch("src.train.build_plan", side_effect=AssertionError("Help read configuration")), contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as error:
            train_main()
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--dry-run", stdout.getvalue())

    def test_shared_run_name_and_device_errors_precede_side_effects(self):
        project = self.root / "runs"
        for name in ("CON", "nul", "../escape", None):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "name"):
                run_directory(project, name)
        original_import = builtins.__import__

        def without_torch(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("Invalid device imported torch")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_torch):
            with self.assertRaisesRegex(ValueError, "device"):
                select_device("cuda:0")
            self.assertEqual(select_device("cpu"), "cpu")
        self.assertFalse(project.exists())
