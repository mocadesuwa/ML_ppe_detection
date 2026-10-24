"""Loopback-only goggles image demo; uploaded bytes stay in memory and never enter training."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .check_head_crops import canonical
from .classification_data import read_json
from .classification_model import completed_training_record, load_classifier, prediction_row
from .common import local_caches, project_path, select_device, sha256

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
CLASSES = ["not_wearing_goggles", "wearing_goggles"]
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}


def decode_image(content: bytes):
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("请选择不超过8MB的静态头像图片。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as opened:
                if opened.format not in {"JPEG", "PNG", "BMP", "WEBP"} or getattr(opened, "n_frames", 1) != 1:
                    raise ValueError("只支持静态JPG、PNG、BMP和WebP，不支持动图。")
                if opened.width * opened.height > MAX_IMAGE_PIXELS or min(opened.size) < 16:
                    raise ValueError("图片尺寸需至少16×16，且总像素不超过1600万。")
                opened.verify()
            with Image.open(io.BytesIO(content)) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("图片无法读取，请使用完整、尺寸合适的静态图片。") from error


class ImageClassifier:
    def __init__(self, weights: Path, requested_device: str):
        self.weights = weights.resolve()
        self.record = completed_training_record(self.weights)
        if self.record.get("parameters", {}).get("classes") != CLASSES or self.record["mode"] != "baseline":
            raise ValueError("演示仅接受已完成的护目镜二分类baseline模型。")
        self.weights_digest = sha256(self.weights)
        self.record_digest = sha256(self.weights.parent / "experiment.json")
        self.requested_device = requested_device
        self.lock = threading.Lock()
        self.network = self.transform = None
        self.known_bytes, self.known_pixels = {}, {}
        data = self.record["parameters"].get("data")
        if isinstance(data, str) and (project_path(data) / "manifest.json").is_file():
            manifest = read_json(project_path(data) / "manifest.json")
            if manifest.get("dataset_fingerprint") != self.record["dataset_fingerprint"]:
                raise ValueError("模型对应的数据清单已变化。")
            for subset, rows in manifest["splits"].items():
                for row in rows:
                    self.known_bytes.setdefault(row["sha256"], set()).add(subset)
                    self.known_pixels.setdefault(row["pixel_sha256"], set()).add(subset)

    def predict(self, content: bytes) -> dict:
        image = decode_image(content)
        size = {"width": image.width, "height": image.height}
        digest = hashlib.sha256(content).hexdigest()
        pixels = hashlib.sha256(canonical(size).encode("utf-8") + b"\0" + image.tobytes()).hexdigest()
        known = sorted(self.known_bytes.get(digest, set()) | self.known_pixels.get(pixels, set()))
        started = time.perf_counter()
        with self.lock:
            if (sha256(self.weights) != self.weights_digest
                    or sha256(self.weights.parent / "experiment.json") != self.record_digest):
                raise ValueError("模型文件或运行记录已变化，请重启演示服务。")
            import torch
            if self.network is None:
                from scripts.preflight_classification import build_transform
                local_caches()
                torch.set_num_threads(2)
                selected = select_device(self.requested_device)
                self.device = "cpu" if selected == "cpu" else f"cuda:{selected}"
                self.network, checkpoint = load_classifier(self.weights, self.device, classes=CLASSES,
                                                           dataset_fingerprint=self.record["dataset_fingerprint"])
                if checkpoint.get("config_sha256") != self.record["config_sha256"] or checkpoint.get("mode") != "baseline":
                    self.network = None
                    raise ValueError("模型检查点与已完成训练记录不一致。")
                self.transform = build_transform()
            with torch.inference_mode():
                logits = self.network(self.transform(image).unsqueeze(0).to(self.device))
                if logits.shape != (1, 2) or not torch.isfinite(logits).all():
                    raise RuntimeError("模型没有返回有效的两类分数。")
                row = prediction_row(logits.softmax(1).cpu().tolist()[0], CLASSES)
        return {**row, "image_size": size, "source_sha256": digest, "known_splits": known,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "weights_sha256": self.weights_digest, "device": self.device,
                "probability_note": "分数未经概率校准；未匹配到已登记头像不代表已证明来源独立。"}


class DemoApplication:
    def __init__(self, weights: Path, candidate: Path | None, device: str):
        self.models = {"baseline": ImageClassifier(weights, device)}
        if candidate:
            self.models["candidate"] = ImageClassifier(candidate, device)

    def metadata(self) -> dict:
        return {"models": [{"id": key, "name": "当前基线" if key == "baseline" else "微调实验模型",
                            "run_name": model.record["parameters"].get("name", model.weights.parent.name),
                            "note": "既往测试结果见项目报告" if key == "baseline" else "仅按验证集选择，尚无新的独立测试"}
                           for key, model in self.models.items()], "max_upload_bytes": MAX_UPLOAD_BYTES}

    def predict(self, model_id: str, content: bytes) -> dict:
        if model_id not in self.models:
            raise ValueError("请选择页面提供的模型。")
        return {"model_id": model_id, **self.models[model_id].predict(content)}


def make_server(application, port: int = 8765) -> ThreadingHTTPServer:
    static_root = project_path("web/dist")

    class Handler(BaseHTTPRequestHandler):
        def allowed_request(self) -> bool:
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in hosts and (not origin or origin in {"http://" + host for host in hosts})

        def respond(self, status: int, content: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(content)

        def json(self, status: int, value: dict) -> None:
            self.respond(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if not self.allowed_request():
                self.json(403, {"error": "仅允许本机同源访问。"})
                return
            if self.path == "/api/models":
                self.json(200, application.metadata())
            elif self.path in STATIC:
                filename, content_type = STATIC[self.path]
                self.respond(200, (static_root / filename).read_bytes(), content_type)
            else:
                self.json(404, {"error": "页面不存在。"})

        def do_POST(self):
            if not self.allowed_request():
                self.json(403, {"error": "仅允许本机同源访问。"})
                return
            if self.path != "/api/predict":
                self.json(404, {"error": "接口不存在。"})
                return
            if self.headers.get("Transfer-Encoding"):
                self.json(400, {"error": "请提交完整的图片文件。"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPLOAD_BYTES:
                    self.json(413, {"error": "请选择不超过8MB的图片。"})
                    return
                self.connection.settimeout(15)
                content = self.rfile.read(length)
                if len(content) != length:
                    raise ValueError("图片上传不完整。")
                result = application.predict(self.headers.get("X-Model-Id", "baseline"), content)
                self.json(200, result)
            except ValueError as error:
                self.json(422, {"error": str(error)})
            except (OSError, RuntimeError, KeyError) as error:
                print(f"Demo prediction failed: {error}", flush=True)
                self.json(500, {"error": "识别失败，请检查服务窗口中的提示。"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--device", default="0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true", help="Open the local page in the default browser")
    args = parser.parse_args()
    server = None
    try:
        if not 0 <= args.port <= 65535:
            raise ValueError("端口需在0到65535之间。")
        application = DemoApplication(project_path(args.weights), project_path(args.candidate) if args.candidate else None, args.device)
        server = make_server(application, args.port)
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Local goggles demo: {url}", flush=True)
        print("Static head images only. Uploads stay in memory; Ctrl+C stops the server.", flush=True)
        if args.open_browser:
            import webbrowser
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    finally:
        if server:
            server.server_close()


if __name__ == "__main__":
    main()
