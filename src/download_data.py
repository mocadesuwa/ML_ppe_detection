"""Download original files from the publisher; preserve the archive and its hash."""
from __future__ import annotations

import argparse
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .common import project_path, sha256, utc_now, write_json

URL = "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/face-mask-detection"


def extract_archive(archive: Path, destination: Path) -> None:
    """Validate all ZIP paths and publish only a fully extracted dataset."""
    def require_empty(path: Path) -> None:
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError(f"Extraction destination must not be a link: {path}")
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"Raw data already exists: {path}")

    require_empty(destination)
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        plan = []
        files, directories = set(), set()
        for entry in zipped.infolist():
            member = entry.filename.replace("\\", "/")
            is_directory = member.endswith("/")
            parts = member.split("/")
            if is_directory:
                parts.pop()
            if any(part in ("", ".", "..") or part.endswith((".", " ")) or any(c in ':<>|"?*' or ord(c) < 32 for c in part) for part in parts):
                raise ValueError(f"Invalid archive path: {entry.filename}")
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or (kind == stat.S_IFDIR and not is_directory):
                raise ValueError(f"Archive links and special files are not accepted: {entry.filename}")
            relative = Path(*parts)
            if not (destination / relative).resolve().is_relative_to(destination):
                raise ValueError(f"Archive entry leaves destination: {entry.filename}")
            # Match paths case-insensitively so archives remain usable on Windows.
            key = tuple(part.casefold() for part in parts)
            parents = [key[:length] for length in range(1, len(key))]
            if any(parent in files for parent in parents) or key in files or (not is_directory and key in directories):
                raise ValueError(f"Conflicting archive paths: {entry.filename}")
            directories.update(parents)
            (directories if is_directory else files).add(key)
            plan.append((entry, relative, is_directory))
        if not files:
            raise ValueError("Archive contains no dataset files")

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".ppe-extract-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "data"
            staged.mkdir()
            for entry, relative, is_directory in plan:
                target = staged / relative
                if is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(entry) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
            require_empty(destination)
            existed = destination.exists()
            if existed:
                destination.rmdir()
            try:
                for attempt in range(5):
                    require_empty(destination)
                    try:
                        staged.rename(destination)
                        break
                    except PermissionError as error:
                        # Windows may briefly retain a handle to a removed
                        # empty directory (e.g. filesystem indexing/scanning).
                        if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 4:
                            raise
                        time.sleep(0.05 * 2 ** attempt)
            except OSError:
                if existed and not destination.exists():
                    destination.mkdir()
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/extract the original Face Mask Detection archive")
    parser.add_argument("--archive", help="Use an already downloaded ZIP instead of accessing the network")
    parser.add_argument("--output", default="dataset/raw/face-mask-detection")
    parser.add_argument("--timeout", type=int, default=30, help="Network inactivity timeout in seconds")
    args = parser.parse_args()
    destination = project_path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Raw data already exists: {destination}")
    archive = project_path(args.archive) if args.archive else destination.parent / "face-mask-detection.zip"
    if not archive.is_file():
        if args.archive:
            raise FileNotFoundError(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".zip.part")
        request = urllib.request.Request(URL, headers={"User-Agent": "PPE-course-baseline/0.1"})
        print(f"Downloading original archive to {archive} ...", flush=True)
        received = 0
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    received += len(chunk)
                    if received % (16 * 1024 * 1024) < len(chunk):
                        print(f"Received {received / 1024 / 1024:.1f} MiB", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RuntimeError("Original download did not complete. Check network access, or download the original ZIP in a browser and use --archive.") from error
        if not zipfile.is_zipfile(temporary):
            raise ValueError("The publisher returned a non-ZIP response. Download from the Kaggle page, then use --archive.")
        temporary.replace(archive)
    extract_archive(archive, destination)
    write_json(destination / "source.json", {"source_url": URL, "download_page": "https://www.kaggle.com/datasets/andrewmvd/face-mask-detection", "retrieved_at": utc_now(), "archive": str(archive), "sha256": sha256(archive), "license_verified": False})
    print(f"Original data extracted to {destination}")


if __name__ == "__main__":
    main()
