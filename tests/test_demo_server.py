"""Real loopback HTTP and image decoding checks; model inference is isolated behind a fake engine."""
import hashlib
import http.client
import io
import json
import threading
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

from src.serve_demo import DemoApplication, MAX_UPLOAD_BYTES, decode_image, make_server


def png_bytes(size=(40, 50)):
    output = io.BytesIO()
    Image.new("RGB", size, "#408c77").save(output, "PNG")
    return output.getvalue()


class ImageDecodeTests(unittest.TestCase):
    def test_valid_image_keeps_pixels_and_exif_is_applied(self):
        image = decode_image(png_bytes())
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (40, 50))
        output = io.BytesIO()
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (24, 40), "white").save(output, "JPEG", exif=exif)
        self.assertEqual(decode_image(output.getvalue()).size, (40, 24))

    def test_corrupt_unsupported_and_animated_images_are_rejected(self):
        animated = io.BytesIO()
        Image.new("RGB", (40, 40), "white").save(animated, "PNG", save_all=True,
                                                  append_images=[Image.new("RGB", (40, 40), "black")], duration=100)
        gif = io.BytesIO()
        Image.new("RGB", (40, 40), "white").save(gif, "GIF")
        for content in (b"", b"not an image", png_bytes()[:30], animated.getvalue(), gif.getvalue()):
            with self.subTest(length=len(content)), self.assertRaises(ValueError):
                decode_image(content)

    def test_dimensions_and_upload_size_are_bounded(self):
        with self.assertRaises(ValueError):
            decode_image(png_bytes((15, 30)))
        with patch("src.serve_demo.MAX_IMAGE_PIXELS", 1000), self.assertRaises(ValueError):
            decode_image(png_bytes())
        with patch("src.serve_demo.MAX_UPLOAD_BYTES", 16), self.assertRaises(ValueError):
            decode_image(png_bytes())


class DemoHttpTests(unittest.TestCase):
    def setUp(self):
        class FakeEngine:
            weights = Path("fixture/best.pt")
            record = {"parameters": {"name": "fixture_model"}}

            def predict(self, content):
                image = decode_image(content)
                return {"label": "wearing_goggles", "class_id": 1, "score": .75,
                        "probabilities": {"not_wearing_goggles": .25, "wearing_goggles": .75},
                        "image_size": {"width": image.width, "height": image.height},
                        "source_sha256": hashlib.sha256(content).hexdigest()}

        app = DemoApplication.__new__(DemoApplication)
        app.models = {"baseline": FakeEngine()}
        self.server = make_server(app, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        self.addCleanup(connection.close)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        content = response.read()
        return response.status, dict(response.getheaders()), content

    def test_page_assets_and_model_metadata_are_served(self):
        for path, content_type in (
            ("/", "text/html"), ("/app.js", "text/javascript"),
            ("/style.css", "text/css"), ("/training.js", "text/javascript"),
            ("/training-summary.json", "application/json"),
        ):
            status, headers, content = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertTrue(headers["Content-Type"].startswith(content_type))
            self.assertTrue(content)
            self.assertEqual(headers["Cache-Control"], "no-store")
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        status, _, content = self.request("GET", "/api/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(content)["models"][0]["id"], "baseline")

    def test_training_summary_separates_validation_from_historical_test(self):
        status, _, content = self.request("GET", "/training-summary.json")
        self.assertEqual(status, 200)
        summary = json.loads(content)
        self.assertFalse(summary["new_training_performed"])
        self.assertFalse(summary["new_test_evaluated"])
        self.assertEqual(summary["historical_test"]["model_id"], "control")
        self.assertEqual(summary["historical_test"]["images"], 24)
        self.assertEqual(sum(summary["counts"]["val"].values()), 47)
        self.assertEqual([run["id"] for run in summary["experiments"]],
                         ["control", "joint", "tail2", "tail4"])
        self.assertEqual([run["id"] for run in summary["experiments"] if run["selected"]],
                         ["control"])

    def test_binary_image_round_trip_uses_selected_model_and_correct_pixels(self):
        image = png_bytes()
        status, _, content = self.request("POST", "/api/predict", image,
                                          {"Content-Type": "image/png", "X-Model-Id": "baseline"})
        self.assertEqual(status, 200)
        result = json.loads(content)
        self.assertEqual(result["image_size"], {"width": 40, "height": 50})
        self.assertEqual(result["source_sha256"], hashlib.sha256(image).hexdigest())
        self.assertEqual(result["model_id"], "baseline")
        self.assertNotIn("detections", result)

    def test_unknown_model_corrupt_image_and_oversized_upload_fail(self):
        for body, headers, expected in ((png_bytes(), {"X-Model-Id": "../weights"}, 422),
                                        (b"corrupt", {}, 422), (b"", {}, 413)):
            status, _, content = self.request("POST", "/api/predict", body, headers)
            self.assertEqual(status, expected)
            self.assertIn("error", json.loads(content))
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        self.addCleanup(connection.close)
        connection.putrequest("POST", "/api/predict")
        connection.putheader("Content-Length", str(MAX_UPLOAD_BYTES + 1))
        connection.endheaders()
        self.assertEqual(connection.getresponse().status, 413)

    def test_cross_origin_and_foreign_host_are_rejected(self):
        for headers in ({"Origin": "https://example.org"}, {"Host": "foreign.example"}):
            status, _, _ = self.request("POST", "/api/predict", png_bytes(), headers)
            self.assertEqual(status, 403)

    def test_static_routes_do_not_expose_repository_or_weights(self):
        for path in ("/.git/config", "/../README.md", "/weights/best.pt", "/dataset/", "/api/unknown"):
            self.assertEqual(self.request("GET", path)[0], 404)


if __name__ == "__main__":
    unittest.main()
