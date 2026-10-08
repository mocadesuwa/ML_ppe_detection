"""Run all regression tests using disposable files under the project cache."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    cache = root / ".cache" / "tests"
    cache.mkdir(parents=True, exist_ok=True)
    previous_tempdir = tempfile.tempdir
    with tempfile.TemporaryDirectory(prefix="regression-", dir=cache) as temporary:
        try:
            tempfile.tempdir = temporary
            suite = unittest.defaultTestLoader.discover(str(root / "tests"))
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        finally:
            tempfile.tempdir = previous_tempdir
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
