"""Integrity and grouped split contracts, using disposable synthetic crops."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import check_head_crops, split_head_crops
from src.common import publish_path, sha256, write_json
from src.prepare_head_crops import build_plan, prepare


class CropIntegrityTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "原图"
        self.raw.mkdir()
        self.manifest = self.root / "heads.json"
        self.output = self.root / "crops"
        self.split_output = self.root / "split_plan"
        self.items = []
        for group in range(6):
            source = self.raw / f"scene_{group}.png"
            image = Image.new("RGB", (8, 6))
            image.putdata([(group * 30, x * 20, y * 30) for y in range(6) for x in range(8)])
            image.save(source)
            for index, label in enumerate(("class_a", "class_b")):
                self.items.append({"id": f"head_{group}_{index}", "source": str(source), "group": f"scene_{group}",
                                   "box_xyxy": [index * 4, 0, (index + 1) * 4, 6], "label": label})

    def prepare(self, items=None):
        write_json(self.manifest, {"format_version": 1, "items": self.items if items is None else items})
        return prepare(build_plan(self.manifest, self.output))

    def plan(self):
        return split_head_crops.build_split(self.output, ["class_a", "class_b"], [0.6, 0.2, 0.2], 42)

    def snapshot(self):
        return {path.relative_to(self.root).as_posix(): sha256(path) for path in self.root.rglob("*") if path.is_file()}

    def rewrite_ledger(self, report):
        write_json(self.output / "crops_manifest.json", report)

    def test_verification_is_read_only_and_does_not_claim_manual_review(self):
        self.prepare()
        before = self.snapshot()
        checked = check_head_crops.inspect_crops(self.output)
        self.assertEqual(checked["crop_count"], 12)
        self.assertEqual(checked["label_counts"], {"class_a": 6, "class_b": 6})
        self.assertEqual(checked["unlabeled_count"], 0)
        self.assertEqual(checked["status"], "integrity_passed")
        self.assertFalse(checked["visual_review_completed"])
        self.assertFalse(checked["training_ready"])
        self.assertEqual(checked, check_head_crops.inspect_crops(self.output))
        self.assertEqual(before, self.snapshot())

    def test_unlabeled_crops_are_valid_but_cannot_be_split(self):
        items = copy.deepcopy(self.items)
        del items[0]["label"]
        self.prepare(items)
        self.assertEqual(check_head_crops.inspect_crops(self.output)["unlabeled_count"], 1)
        with self.assertRaisesRegex(ValueError, "explicit label"):
            self.plan()
        self.assertFalse(self.split_output.exists())

    def test_record_tampering_and_duplicate_json_keys_are_rejected(self):
        report = self.prepare()
        ledger = self.output / "crops_manifest.json"
        for field, value in (("output_file", "../escape.png"), ("group", "different"),
                             ("box_xyxy", [0, 0, 3, 6]), ("label", "class_b"),
                             ("crop_size", {"width": True, "height": 6})):
            changed = copy.deepcopy(report)
            changed["crops"][0][field] = value
            self.rewrite_ledger(changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_head_crops.inspect_crops(self.output)
        ledger.write_text('{"format_version":1,"format_version":1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Repeated"):
            check_head_crops.inspect_crops(self.output)

    def test_incomplete_and_invalid_ledger_schema_are_rejected(self):
        report = self.prepare()
        for key, value in (("format_version", True), ("purpose", "other"), ("status", "running"),
                           ("sources", []), ("crops", []), ("visual_review_completed", True), ("created_at", None)):
            self.rewrite_ledger({**report, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_head_crops.inspect_crops(self.output)
        self.rewrite_ledger({**report, "unknown": 1})
        with self.assertRaises(ValueError):
            check_head_crops.inspect_crops(self.output)

    def test_missing_and_unexpected_files_are_rejected(self):
        self.prepare()
        path = self.output / "images/head_0_0.png"
        content = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            check_head_crops.inspect_crops(self.output)
        path.write_bytes(content)
        unexpected = self.output / "images/orphan.txt"
        unexpected.write_text("orphan")
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            check_head_crops.inspect_crops(self.output)

    def test_modified_file_hash_and_rehashed_wrong_pixels_are_rejected(self):
        from PIL import Image

        report = self.prepare()
        path = self.output / "images/head_0_0.png"
        Image.new("RGB", (4, 6), "black").save(path)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            check_head_crops.inspect_crops(self.output)
        report["crops"][0]["output_sha256"] = sha256(path)
        self.rewrite_ledger(report)
        with self.assertRaisesRegex(ValueError, "pixels differ"):
            check_head_crops.inspect_crops(self.output)

    def test_rehashed_wrong_size_or_mode_are_rejected(self):
        from PIL import Image

        report = self.prepare()
        path = self.output / "images/head_0_0.png"
        for mode, size in (("L", (4, 6)), ("RGB", (3, 6))):
            Image.new(mode, size).save(path)
            report["crops"][0]["output_sha256"] = sha256(path)
            self.rewrite_ledger(report)
            with self.subTest(mode=mode, size=size), self.assertRaisesRegex(ValueError, "format/size"):
                check_head_crops.inspect_crops(self.output)

    def test_changed_original_manifest_or_image_invalidates_verification(self):
        self.prepare()
        content = self.manifest.read_bytes()
        self.manifest.write_bytes(content + b"\n")
        with self.assertRaisesRegex(ValueError, "provenance"):
            check_head_crops.inspect_crops(self.output)
        self.manifest.write_bytes(content)
        source = Path(self.items[0]["source"])
        source.write_bytes(source.read_bytes() + b"modified")
        with self.assertRaisesRegex(ValueError, "provenance"):
            check_head_crops.inspect_crops(self.output)

    def test_changes_during_verification_are_detected(self):
        self.prepare()
        real_check = check_head_crops.require_unchanged_inputs

        def mutate(plan):
            real_check(plan)
            path = self.output / "images/head_0_0.png"
            path.write_bytes(path.read_bytes() + b"changed")

        with patch("src.check_head_crops.require_unchanged_inputs", side_effect=mutate), self.assertRaisesRegex(ValueError, "changed during"):
            check_head_crops.inspect_crops(self.output)

    def test_grouped_splits_are_reproducible_complete_and_label_balanced(self):
        self.prepare()
        before = self.snapshot()
        first, second = self.plan(), self.plan()
        self.assertEqual(first["splits"], second["splits"])
        self.assertEqual(first["crop_fingerprint"], second["crop_fingerprint"])
        seen, groups = set(), {}
        for split, items in first["splits"].items():
            self.assertEqual({item["label"] for item in items}, {"class_a", "class_b"})
            for item in items:
                self.assertNotIn(item["id"], seen)
                seen.add(item["id"])
                for key in ("source_group", "split_group"):
                    group = (key, item[key])
                    self.assertEqual(groups.setdefault(group, split), split)
        self.assertEqual(seen, {item["id"] for item in self.items})
        self.assertEqual(first["independent_groups"], 6)
        self.assertFalse(first["training_ready"])
        self.assertFalse(first["visual_review_completed"])
        self.assertEqual(before, self.snapshot())

    def test_identical_pixels_join_different_source_groups_transitively(self):
        from PIL import Image

        # A larger source has different bytes but an identical first crop.
        source = self.raw / "copied_region.png"
        with Image.open(self.items[0]["source"]) as image:
            larger = Image.new("RGB", (10, 6), "white")
            larger.paste(image, (0, 0))
            larger.save(source)
        extra = {**self.items[0], "id": "head_duplicate", "source": str(source), "group": "another_scene"}
        self.prepare(self.items + [extra])
        plan = self.plan()
        records = {item["id"]: (split, item) for split, items in plan["splits"].items() for item in items}
        self.assertEqual(records["head_0_0"][0], records["head_duplicate"][0])
        self.assertEqual(records["head_0_1"][1]["split_group"], records["head_duplicate"][1]["split_group"])
        self.assertEqual(plan["independent_groups"], 6)

    def test_identical_pixels_with_conflicting_labels_fail(self):
        extra = {**self.items[0], "id": "conflict", "label": "class_b"}
        self.prepare(self.items + [extra])
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            self.plan()

    def test_explicit_classes_and_split_parameters_are_validated(self):
        self.prepare()
        for classes in ([], ["class_a"], ["class_a", "class_b", "absent"], ["class_a", "class_a"], [True], [" class_a", "class_b"]):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                split_head_crops.build_split(self.output, classes, [0.6, 0.2, 0.2], 42)
        for ratios, seed in (([1, 0, 0], 42), ([0.6, 0.2, 0.1], 42), ([float("nan"), 0.2, 0.2], 42), ([0.6, 0.2, 0.2], True)):
            with self.subTest(ratios=ratios, seed=seed), self.assertRaises(ValueError):
                split_head_crops.build_split(self.output, ["class_a", "class_b"], ratios, seed)

    def test_insufficient_independent_groups_fail_without_partial_split(self):
        self.prepare([{**item, "group": "same_session"} for item in self.items])
        with self.assertRaisesRegex(ValueError, "three independent groups"):
            self.plan()
        self.assertFalse(self.split_output.exists())

    def test_publish_only_writes_a_new_plan_and_preserves_inputs(self):
        self.prepare()
        before = self.snapshot()
        plan = self.plan()
        split_head_crops.publish_split(plan, self.split_output)
        actual = json.loads((self.split_output / "splits_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(actual, plan)
        after = self.snapshot()
        self.assertEqual({key: after[key] for key in before}, before)
        self.assertEqual(set(after) - set(before), {"split_plan/splits_manifest.json"})
        with self.assertRaises(FileExistsError):
            split_head_crops.publish_split(plan, self.split_output)

    def test_outputs_inside_inputs_and_stale_or_tampered_plans_are_rejected(self):
        self.prepare()
        plan = self.plan()
        for directory in (self.raw / "splits", self.output / "splits"):
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "outside"):
                split_head_crops.publish_split(plan, directory)
            self.assertFalse(directory.exists())
        changed = copy.deepcopy(plan)
        changed["splits"]["train"].pop()
        with self.assertRaisesRegex(ValueError, "changed before"):
            split_head_crops.publish_split(changed, self.split_output)
        report = json.loads((self.output / "crops_manifest.json").read_text(encoding="utf-8"))
        report["created_at"] = "changed evidence"
        self.rewrite_ledger(report)
        with self.assertRaisesRegex(ValueError, "changed before"):
            split_head_crops.publish_split(plan, self.split_output)
        self.assertFalse(self.split_output.exists())

    def test_failed_write_and_changed_inputs_do_not_publish_split(self):
        self.prepare()
        plan = self.plan()
        with patch("src.split_head_crops.write_json", side_effect=OSError("Synthetic write failure")), self.assertRaises(OSError):
            split_head_crops.publish_split(plan, self.split_output)
        self.assertEqual(list(self.root.glob(".head-crops-*")), [])
        real_write = split_head_crops.write_json

        def mutate(path, value):
            real_write(path, value)
            ledger = self.output / "crops_manifest.json"
            ledger.write_bytes(ledger.read_bytes() + b"\n")

        with patch("src.split_head_crops.write_json", side_effect=mutate), self.assertRaisesRegex(ValueError, "changed during"):
            split_head_crops.publish_split(plan, self.split_output)
        self.assertFalse(self.split_output.exists())
        self.assertEqual(list(self.root.glob(".head-crops-*")), [])

    def test_cli_check_and_split_dry_run_do_not_write_or_load_models(self):
        self.prepare()
        before = self.snapshot()
        commands = (
            (check_head_crops.main, ["check", "--crops", str(self.output)]),
            (split_head_crops.main, ["split", "--crops", str(self.output), "--classes", "class_a", "class_b", "--output", str(self.split_output), "--dry-run"]),
        )
        for main, argv in commands:
            stdout = io.StringIO()
            with patch("sys.argv", argv), patch.dict("sys.modules", {"torch": None, "ultralytics": None}), contextlib.redirect_stdout(stdout):
                main()
            self.assertFalse(json.loads(stdout.getvalue())["training_ready"])
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.split_output.exists())

    def test_cli_errors_have_nonzero_exit_without_traceback(self):
        self.prepare()
        argv = ["split", "--crops", str(self.output), "--classes", "class_a", "--output", str(self.split_output)]
        stderr = io.StringIO()
        with patch("sys.argv", argv), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            split_head_crops.main()
        self.assertEqual(error.exception.code, 2)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertFalse(self.split_output.exists())

    def test_windows_publication_retries_are_bounded_and_do_not_copy(self):
        error = PermissionError("Synthetic Windows sharing error")
        error.winerror = 5
        staged = self.root / "staged"
        staged.mkdir()
        real_rename = Path.rename
        calls = []

        def rename(path, target):
            calls.append(target)
            if len(calls) == 1:
                raise error
            return real_rename(path, target)

        with patch.object(Path, "rename", rename), patch("src.common.time.sleep") as sleep:
            publish_path(staged, self.split_output)
        self.assertEqual(len(calls), 2)
        sleep.assert_called_once_with(0.05)
        self.assertTrue(self.split_output.is_dir())
        self.assertFalse(staged.exists())

    def test_permanent_and_non_windows_publication_errors_are_not_hidden(self):
        error = PermissionError("Synthetic permanent access error")
        error.winerror = 5
        for failure, attempts in ((error, 6), (PermissionError("Other permission error"), 1)):
            with self.subTest(failure=failure), patch.object(Path, "rename", side_effect=failure) as rename, patch("src.common.time.sleep"), self.assertRaises(PermissionError):
                publish_path(self.root / "staged", self.split_output)
            self.assertEqual(rename.call_count, attempts)
        self.assertFalse(self.split_output.exists())

    def test_publication_retry_does_not_overwrite_an_appearing_output(self):
        error = PermissionError("Synthetic sharing error")
        error.winerror = 32

        def appeared(delay):
            self.split_output.mkdir()
            (self.split_output / "keep.txt").write_text("keep")

        with patch.object(Path, "rename", side_effect=error) as rename, patch("src.common.time.sleep", side_effect=appeared), self.assertRaises(FileExistsError):
            publish_path(self.root / "staged", self.split_output)
        self.assertEqual(rename.call_count, 1)
        self.assertEqual((self.split_output / "keep.txt").read_text(), "keep")

    def test_atomic_json_replace_retries_windows_errors(self):
        path = self.root / "report.json"
        write_json(path, {"version": 1})
        real_replace = Path.replace
        calls = []
        error = PermissionError("Synthetic sharing error")
        error.winerror = 33

        def replace(staged, target):
            calls.append(target)
            if len(calls) == 1:
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 1})
                raise error
            return real_replace(staged, target)

        with patch.object(Path, "replace", replace), patch("src.common.time.sleep"):
            write_json(path, {"version": 2})
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 2})
        self.assertFalse(path.with_suffix(".json.part").exists())

    def test_failed_json_replace_keeps_previous_complete_record(self):
        path = self.root / "report.json"
        write_json(path, {"version": 1})
        error = PermissionError("Synthetic permanent access error")
        error.winerror = 5
        with patch.object(Path, "replace", side_effect=error) as replace, patch("src.common.time.sleep"), self.assertRaises(PermissionError):
            write_json(path, {"version": 2})
        self.assertEqual(replace.call_count, 6)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 1})
        self.assertTrue(path.with_suffix(".json.part").exists())


if __name__ == "__main__":
    unittest.main()
