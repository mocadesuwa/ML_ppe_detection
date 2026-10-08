"""Download original files from the publisher; preserve the archive and its hash."""
from __future__ import annotations

import argparse
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .common import project_path, sha256, utc_now, write_json

URL = "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/face-mask-detection"


def extract_archive(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            member = entry.filename.replace("\\", "/")
            target = (destination / member).resolve()
            if not target.is_relative_to(destination) or Path(member).is_absolute() or ":" in member:
                raise ValueError(f"Archive entry leaves destination: {entry.filename}")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not accepted")
        zipped.extractall(destination)


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
