"""Task selection and local import must be explicit and preserve source content."""
import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src import check_dataset, evaluate, import_data, predict, prepare_dataset, review_dataset, source_review, train
from src.common import sha256


class TaskCleanupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_cli_requires_explicit_task_inputs_before_any_work(self):
        cases = (
            (train.main, ["train", "--dry-run"], "--config"),
            (prepare_dataset.main, ["prepare"], "--config"),
            (check_dataset.main, ["check"], "--data"),
            (review_dataset.main, ["review"], "--data"),
            (evaluate.main, ["evaluate"], "--weights"),
            (predict.main, ["predict", "--source", "head.png"], "--weights"),
            (import_data.main, ["import"], "--archive"),
            (source_review.main, ["audit", "--raw", "source", "--output", "audit"], "--classes"),
        )
        for main, argv, required in cases:
            stderr = io.StringIO()
            with self.subTest(command=argv[0]), patch("sys.argv", argv), patch("src.train.build_plan") as plan, contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn(required, stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            plan.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def invoke_import(self, archive, output):
        argv = ["import", "--archive", str(archive), "--output", str(output), "--source-url", "https://example.org/reviewed-source"]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            import_data.main()

    def test_import_preserves_archive_and_embedded_source_record(self):
        archive, output = self.root / "source.zip", self.root / "raw"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("images/sample.txt", "original sample")
            zipped.writestr("source.json", '{"original": true}')
        before = archive.read_bytes()
        self.invoke_import(archive, output)
        self.assertEqual(archive.read_bytes(), before)
        self.assertEqual((output / "source.json").read_text(), '{"original": true}')
        record = json.loads((self.root / "raw.source.json").read_text(encoding="utf-8"))
        self.assertEqual(record["sha256"], sha256(archive))
        self.assertEqual(record["source_url"], "https://example.org/reviewed-source")
        self.assertFalse(record["license_verified"])

    def test_existing_import_record_is_not_overwritten(self):
        archive, output = self.root / "source.zip", self.root / "raw"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("sample.txt", "original")
        record = self.root / "raw.source.json"
        record.write_text("keep old evidence", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.invoke_import(archive, output)
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(record.read_text(), "keep old evidence")
        self.assertFalse(output.exists())
