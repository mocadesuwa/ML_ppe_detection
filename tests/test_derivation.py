"""Reviewed source repairs must be explicit, bound to content, and reversible."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.derive_dataset import derive
from scripts.review_raw_dataset import inspect, snapshot
from src.common import write_json


class DerivationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        (self.raw / "images").mkdir(parents=True)
        (self.raw / "annotations").mkdir()
        from PIL import Image
        for name, xmax in (("a", 11), ("b", 7), ("c", 6)):
            Image.new("RGB", (10, 12), (50, 70, 90)).save(self.raw / "images" / f"{name}.png")
            (self.raw / "annotations" / f"{name}.xml").write_text(
                f"<annotation><filename>{name}.png</filename><size><width>10</width><height>12</height></size>"
                f"<object><name>with_mask</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>{xmax}</xmax><ymax>7</ymax></bndbox></object></annotation>", encoding="utf-8")
        self.records = {r["filename"]: r for r in inspect(self.raw)["records"]}
        self.decisions = {"format_version": 1, "source_fingerprint": snapshot(self.raw)[1], "coordinates": "voc_one_based_inclusive",
                          "box_repairs": [self.entry("a") | {"object_index": 1, "before": [1, 2, 11, 7], "after": [1, 2, 10, 7]}],
                          "excluded_images": [], "duplicate_removals": [self.entry("c") | {"retained": "b.png"}]}
        self.path = self.root / "decisions.json"
        self.output = self.root / "derived"

    def entry(self, name):
        record = self.records[f"{name}.png"]
        return {k: record[k] for k in ("filename", "image_sha256", "annotation_sha256")} | {"reason": "reviewed synthetic decision"}

    def run_derivation(self):
        write_json(self.path, self.decisions)
        return derive(self.raw, self.output, self.path)

    def test_repair_and_deduplicate_copy_preserves_raw_and_records_hashes(self):
        before = snapshot(self.raw)
        ledger = self.run_derivation()
        self.assertEqual(snapshot(self.raw), before)
        self.assertEqual(ledger["retained_images"], 2)
        self.assertTrue(ledger["box_repairs"][0]["applied"])
        self.assertEqual(inspect(self.output)["invalid_boxes"], [])
        self.assertFalse((self.output / "images/c.png").exists())
        self.assertNotEqual(ledger["selected_files"][0]["original_annotation_sha256"], ledger["selected_files"][0]["derived_annotation_sha256"])
        self.assertEqual(ledger["status"], "complete")

    def test_stale_snapshot_or_entry_hash_stops_before_any_output(self):
        for field in ("source_fingerprint", "annotation_sha256"):
            with self.subTest(field=field):
                if field == "source_fingerprint":
                    old = self.decisions[field]
                    self.decisions[field] = "bad"
                else:
                    old = self.decisions["box_repairs"][0][field]
                    self.decisions["box_repairs"][0][field] = "bad"
                with self.assertRaises(ValueError):
                    self.run_derivation()
                self.assertFalse(self.output.exists())
                if field == "source_fingerprint": self.decisions[field] = old
                else: self.decisions["box_repairs"][0][field] = old

    def test_unlisted_bad_box_and_wrong_repair_are_not_silently_clipped(self):
        repair = self.decisions["box_repairs"]
        for replacement in ([], [repair[0] | {"after": [0, 2, 10, 7]}], [repair[0] | {"before": [2, 2, 11, 7]}]):
            self.decisions["box_repairs"] = replacement
            with self.assertRaises(ValueError): self.run_derivation()
            self.assertFalse(self.output.exists())

    def test_excluded_image_is_removed_whole_and_repair_is_marked_unapplied(self):
        self.decisions["excluded_images"] = [self.entry("a")]
        ledger = self.run_derivation()
        self.assertFalse((self.output / "images/a.png").exists())
        self.assertFalse((self.output / "annotations/a.xml").exists())
        self.assertFalse(ledger["box_repairs"][0]["applied"])

    def test_removed_keeper_and_nonidentical_images_are_rejected(self):
        self.decisions["excluded_images"] = [self.entry("b")]
        with self.assertRaises(ValueError): self.run_derivation()
        self.decisions["excluded_images"] = []
        from PIL import Image
        Image.new("RGB", (10, 12), "red").save(self.raw / "images/c.png")
        self.decisions["source_fingerprint"] = snapshot(self.raw)[1]
        self.decisions["duplicate_removals"][0]["image_sha256"] = inspect(self.raw)["records"][2]["image_sha256"]
        with self.assertRaisesRegex(ValueError, "identical"): self.run_derivation()
        self.assertFalse(self.output.exists())

    def test_existing_output_and_original_directory_are_protected(self):
        write_json(self.path, self.decisions)
        for output in (self.raw, self.raw / "nested", self.raw.parent):
            with self.assertRaises(ValueError): derive(self.raw, output, self.path)
        self.run_derivation()
        before = snapshot(self.output)
        with self.assertRaises(FileExistsError): derive(self.raw, self.output, self.path)
        self.assertEqual(snapshot(self.output), before)


if __name__ == "__main__":
    unittest.main()
