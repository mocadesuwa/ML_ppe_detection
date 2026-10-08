"""Inspect the current interpreter; no packages or drivers are installed here."""
from __future__ import annotations

import argparse
import importlib.metadata
import platform
import subprocess
import sys

from .common import project_path, utc_now, write_json


def inspect_environment() -> dict:
    report = {"checked_at": utc_now(), "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": {}}
    for package in ("torch", "torchvision", "ultralytics", "Pillow", "PyYAML"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=15, check=True)
        report["nvidia_smi"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        report["nvidia_smi_error"] = str(error)
    try:
        import torch
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if report["cuda_available"]:
            tensor = torch.ones((16, 16), device="cuda")
            report["gpu_operation_ok"] = float((tensor @ tensor).sum().item()) == 4096.0
    except (ImportError, RuntimeError, OSError) as error:
        report["torch_error"] = str(error)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Python, dependencies and CUDA readiness")
    parser.add_argument("--output", default="results/environment.json")
    args = parser.parse_args()
    report = inspect_environment()
    write_json(project_path(args.output), report)
    print(f"Environment report: {project_path(args.output)}")
    print(f"CUDA available: {report.get('cuda_available', False)}")


if __name__ == "__main__":
    main()
