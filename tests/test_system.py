"""
System and End-to-End API Tests for AquaSeg Multispectral Water Segmentation.
Uses the Flask test client to validate complete client-server interactions:
1. Web UI delivery (HTML, titles, removal of footer and copy button)
2. Telemetry and Catalog APIs (/health, /api/health, /api/sample_images)
3. Prediction route happy paths (JSON, direct image/png mask, direct overlay/png, model switching, threshold tuning)
4. Comprehensive edge-case and negative input handling:
   - Missing payload -> 400 Bad Request
   - Non-existent sample ID -> 404 Not Found
   - Empty file upload (0 bytes) -> 400 Bad Request
   - Unsupported file extensions (.pdf, .txt, .csv) -> 400 Bad Request
   - Corrupted TIFF files -> 400 Bad Request
   - Non-12-channel TIFF rasters (e.g. 3-channel RGB) -> 400 Bad Request
   - Unknown model architecture -> 400 Bad Request
   - Invalid threshold (non-numeric, negative, > 1.0) -> 400 Bad Request
   - Invalid output format -> 400 Bad Request
   - Unsupported HTTP methods (GET/DELETE /predict) -> 405 Method Not Allowed
   - Unknown URLs -> 404 Not Found
"""

import sys
import unittest
import io
import json
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image

# Add paths
deployment_dir = Path(__file__).resolve().parent.parent
project_root = deployment_dir.parent
for p in [str(deployment_dir), str(project_root), str(project_root / "water_segmentation"), str(project_root / "water_segmentation_part2")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app import app


class TestSystemWebUI(unittest.TestCase):
    """System tests for frontend delivery and HTML structural integrity."""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_index_page_loads(self):
        """Root GET / must return 200 OK and match updated layout requirements."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.content_type)
        html = res.get_data(as_text=True)

        # Title requirement
        self.assertIn("Week 4 water segmentation  deployment with Flask", html)
        self.assertIn("Sensor Specs", html)

        # Removed elements verification
        self.assertNotIn("AquaSeg — Satellite Water Delineation", html)
        self.assertNotIn("btn-copy", html)
        self.assertNotIn('<footer class="footer">', html)

        # Retained functional components
        self.assertIn('id="model-select"', html)
        self.assertIn('id="threshold-slider"', html)
        self.assertIn('id="drop-area"', html)
        self.assertIn('id="btn-run"', html)
        self.assertIn('id="modal-sensor"', html)


class TestSystemHealthAPI(unittest.TestCase):
    """System tests for health check and sample catalog endpoints."""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_health_endpoint(self):
        """GET /health must return status healthy, device string, and model list."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("device", data)
        self.assertIn("available_models", data)
        self.assertIn("se_res_unet", data["available_models"])

    def test_api_health_alias(self):
        """GET /api/health must return identical 200 JSON."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")

    def test_sample_images_catalog(self):
        """GET /api/sample_images must return 5 verified test scenes."""
        res = self.client.get("/api/sample_images")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("samples", data)
        samples = data["samples"]
        self.assertEqual(len(samples), 5)
        ids = [s["id"] for s in samples]
        self.assertEqual(ids, ["3", "8", "26", "48", "70"])


class TestSystemPredictSuccessCases(unittest.TestCase):
    """System tests for valid prediction workflows."""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.sample_dir = deployment_dir / "sample_inputs"

    def test_predict_sample_id_form(self):
        """POST /predict with sample_id in form data returns complete JSON metadata."""
        res = self.client.post("/predict", data={"sample_id": "26"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["filename"], "sample_26.tif")
        self.assertEqual(data["model_key"], "se_res_unet")
        self.assertGreater(data["water_percentage"], 10.0)
        self.assertEqual(data["total_pixels"], 16384)
        self.assertEqual(data["water_pixels"] + data["land_pixels"], 16384)
        self.assertTrue(data["mask_image"].startswith("data:image/png;base64,"))
        self.assertTrue(data["rgb_preview"].startswith("data:image/png;base64,"))
        self.assertTrue(data["overlay_image"].startswith("data:image/png;base64,"))

    def test_predict_sample_id_json_body(self):
        """POST /predict with sample_id in JSON payload returns 200."""
        res = self.client.post(
            "/predict",
            data=json.dumps({"sample_id": "8"}),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["filename"], "sample_8.tif")

    def test_predict_multipart_image_upload(self):
        """POST /predict with multipart file under key 'image'."""
        with open(self.sample_dir / "3.tif", "rb") as f:
            file_bytes = f.read()

        data = {
            "image": (io.BytesIO(file_bytes), "river_scene.tif")
        }
        res = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        resp_data = res.get_json()
        self.assertEqual(resp_data["status"], "success")
        self.assertEqual(resp_data["filename"], "river_scene.tif")

    def test_predict_multipart_file_key_alternate(self):
        """POST /predict with multipart file under key 'file'."""
        with open(self.sample_dir / "3.tif", "rb") as f:
            file_bytes = f.read()

        data = {
            "file": (io.BytesIO(file_bytes), "upload.tif")
        }
        res = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "success")

    def test_predict_format_image_binary_stream(self):
        """POST /predict with format='image' returns image/png binary attachment."""
        res = self.client.post("/predict", data={"sample_id": "3", "format": "image"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content_type, "image/png")
        self.assertIn("attachment", res.headers.get("Content-Disposition", ""))

        # Verify PNG magic bytes
        png_bytes = res.get_data()
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

        # Verify PIL can decode it as a 128x128 mask
        pil_img = Image.open(io.BytesIO(png_bytes))
        self.assertEqual(pil_img.size, (128, 128))

    def test_predict_format_overlay_binary_stream(self):
        """POST /predict with format='overlay' returns image/png RGB composite overlay."""
        res = self.client.post("/predict", data={"sample_id": "3", "format": "overlay"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content_type, "image/png")
        png_bytes = res.get_data()
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_predict_model_switching(self):
        """POST /predict dynamically switches architecture based on 'model' parameter."""
        for m_key in ["se_res_unet", "resnet50_deeplabv3p", "resnet34_unet"]:
            res = self.client.post("/predict", data={"sample_id": "26", "model": m_key})
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["model_key"], m_key)

    def test_predict_threshold_tuning(self):
        """POST /predict updates decision boundary with 'threshold' parameter."""
        res = self.client.post("/predict", data={"sample_id": "26", "threshold": "0.65"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["threshold"], 0.65)


class TestSystemPredictNegativeAndEdgeCases(unittest.TestCase):
    """System tests verifying robust error handling and HTTP status codes for all edge cases."""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_no_input_provided(self):
        """POST /predict with neither image nor sample_id must return 400 Bad Request."""
        res = self.client.post("/predict", data={})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    def test_nonexistent_sample_id(self):
        """POST /predict with non-existent sample_id must return 404 Not Found."""
        res = self.client.post("/predict", data={"sample_id": "999999"})
        self.assertEqual(res.status_code, 404)
        data = res.get_json()
        self.assertIn("error", data)

    def test_empty_file_upload(self):
        """POST /predict with 0-byte file must return 400 Bad Request."""
        data = {
            "image": (io.BytesIO(b""), "empty.tif")
        }
        res = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("empty", data["error"].lower())

    def test_unsupported_file_extension(self):
        """POST /predict with non-tif file (e.g. .pdf, .txt, .csv) must return 400 Bad Request."""
        for bad_ext in ["document.pdf", "data.txt", "table.csv", "photo.png"]:
            data = {
                "image": (io.BytesIO(b"dummy data"), bad_ext)
            }
            res = self.client.post("/predict", data=data, content_type="multipart/form-data")
            self.assertEqual(res.status_code, 400)
            self.assertIn("Unsupported file extension", res.get_json()["error"])

    def test_corrupted_tiff_file(self):
        """POST /predict with random garbage bytes named .tif must return 400 Bad Request."""
        garbage_bytes = b"CORRUPTED_NOT_A_REAL_TIFF_FILE_GARBAGE_BYTES_12345"
        data = {
            "image": (io.BytesIO(garbage_bytes), "corrupted.tif")
        }
        res = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertIn("status", res.get_json())
        self.assertEqual(res.get_json()["status"], "error")

    def test_invalid_channel_count_tiff(self):
        """POST /predict with 3-channel RGB TIFF must return 400 Bad Request explaining 12 channels required."""
        rgb_3ch = np.zeros((128, 128, 3), dtype=np.uint8)
        bio = io.BytesIO()
        tifffile.imwrite(bio, rgb_3ch)
        bio.seek(0)

        data = {
            "image": (bio, "rgb_3channel.tif")
        }
        res = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("Expected 12-channel multispectral image", data["message"])

    def test_unknown_model_name(self):
        """POST /predict with unknown model name must return 400 Bad Request."""
        res = self.client.post("/predict", data={"sample_id": "3", "model": "fake_gpt_v9"})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("Unknown model architecture", data["error"])

    def test_non_numeric_threshold(self):
        """POST /predict with non-numeric threshold must return 400 Bad Request."""
        res = self.client.post("/predict", data={"sample_id": "3", "threshold": "invalid_string"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid threshold value", res.get_json()["error"])

    def test_negative_threshold(self):
        """POST /predict with negative threshold must return 400 Bad Request."""
        res = self.client.post("/predict", data={"sample_id": "3", "threshold": "-0.5"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Decision threshold must be between 0.0 and 1.0", res.get_json()["error"])

    def test_overflow_threshold(self):
        """POST /predict with threshold > 1.0 must return 400 Bad Request."""
        res = self.client.post("/predict", data={"sample_id": "3", "threshold": "1.8"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Decision threshold must be between 0.0 and 1.0", res.get_json()["error"])

    def test_invalid_format(self):
        """POST /predict with invalid format (e.g. mp4, xml) must return 400 Bad Request."""
        res = self.client.post("/predict", data={"sample_id": "3", "format": "mp4"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid format", res.get_json()["error"])

    def test_unsupported_http_methods(self):
        """GET /predict and DELETE /predict must return 405 Method Not Allowed."""
        res_get = self.client.get("/predict")
        self.assertEqual(res_get.status_code, 405)
        self.assertEqual(res_get.get_json()["status"], 405)

        res_del = self.client.delete("/predict")
        self.assertEqual(res_del.status_code, 405)
        self.assertEqual(res_del.get_json()["status"], 405)

    def test_nonexistent_route_404(self):
        """Arbitrary non-existent endpoint must return 404 Not Found."""
        res = self.client.get("/api/this_endpoint_does_not_exist")
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.get_json()["status"], 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
