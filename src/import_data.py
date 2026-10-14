"""Import a local ZIP without changing its contents or accessing the network."""
from __future__ import annotations

import argparse
import shutil
import stat
import tempfile
import time
import zipfile
from pathlib import Path

from .common import config_path, project_path, sha256, utc_now, write_json


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Existing local ZIP")
    parser.add_argument("--output", required=True, help="New or empty directory for original files")
    parser.add_argument("--source-url", required=True, help="Publisher page, recorded as provenance only")
    args = parser.parse_args()
    try:
        archive = config_path(args.archive, "archive", project_path("."))
        destination = config_path(args.output, "output", project_path("."))
        if not args.source_url.strip():
            raise ValueError("source-url must be a non-empty publisher URL")
        provenance = destination.with_name(destination.name + ".source.json")
        if provenance.exists() or provenance.is_symlink():
            raise FileExistsError(f"Source record already exists: {provenance}")
        digest = sha256(archive)
        extract_archive(archive, destination)
        if sha256(archive) != digest:
            raise ValueError("Archive changed during import; output is unconfirmed")
        write_json(provenance, {"source_url": args.source_url.strip(), "imported_at": utc_now(), "archive": str(archive), "sha256": digest, "license_verified": False})
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(f"Original data extracted to {destination}")


if __name__ == "__main__":
    main()
