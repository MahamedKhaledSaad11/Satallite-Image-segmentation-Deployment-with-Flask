"""
Integration Tests for AquaSeg Multispectral Water Segmentation.
Validates end-to-end integration across components:
1. Model loading, checkpoint validation, parameter counts, and caching behavior
2. GeoTIFF multi-channel file parsing from disk and in-memory BytesIO
3. Pipeline execution across 5 benchmark scenes (River, Lake, Coast, Wetland, Arid Control)
4. Decision threshold sensitivity verification (Monotonic water area behavior)
5. Edge cases: All-zero rasters, extreme saturation, and missing/corrupted checkpoints
"""

import sys
import unittest
import io
from pathlib import Path
import numpy as np
import tifffile
import torch

# Add paths
deployment_dir = Path(__file__).resolve().parent.parent
project_root = deployment_dir.parent
for p in [str(deployment_dir), str(project_root), str(project_root / "water_segmentation"), str(project_root / "water_segmentation_part2")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference import InferenceEngine


class TestModelLoadingIntegration(unittest.TestCase):
    """Integration tests for model loading and checkpoint management."""

    @classmethod
    def setUpClass(cls):
        cls.engine = InferenceEngine(default_model="se_res_unet")

    def test_load_all_registered_models(self):
        """All three models must load from checkpoints without errors."""
        for model_key in ["se_res_unet", "resnet50_deeplabv3p", "resnet34_unet"]:
            model = self.engine.load_model(model_key)
            self.assertIsInstance(model, torch.nn.Module)
            self.assertFalse(model.training)  # Must be in eval mode

    def test_model_caching(self):
        """Requesting the same model multiple times must return the cached instance."""
        m1 = self.engine.load_model("se_res_unet")
        m2 = self.engine.load_model("se_res_unet")
        self.assertIs(m1, m2)

    def test_model_parameter_contracts(self):
        """Verify parameter counts match expected architectures."""
        se_res = self.engine.load_model("se_res_unet")
        params_se_res = sum(p.numel() for p in se_res.parameters())
        self.assertGreater(params_se_res, 7_000_000)
        self.assertLess(params_se_res, 10_000_000)  # ~8.12M params

        deeplab = self.engine.load_model("resnet50_deeplabv3p")
        params_deeplab = sum(p.numel() for p in deeplab.parameters())
        self.assertGreater(params_deeplab, 20_000_000)  # ~26.7M params

    def test_invalid_model_key_raises_error(self):
        """Requesting an unregistered model key must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.engine.load_model("non_existent_model_v9")
        self.assertIn("Unknown model key", str(ctx.exception))

    def test_missing_checkpoint_raises_file_not_found(self):
        """If a checkpoint file is missing, must raise FileNotFoundError."""
        original_path = self.engine.MODEL_CONFIGS["se_res_unet"]["checkpoint"]
        try:
            self.engine.MODEL_CONFIGS["se_res_unet"]["checkpoint"] = Path("non_existent_weights.pt")
            # Force reload by removing from cache
            self.engine.models.pop("se_res_unet", None)
            with self.assertRaises(FileNotFoundError):
                self.engine.load_model("se_res_unet")
        finally:
            self.engine.MODEL_CONFIGS["se_res_unet"]["checkpoint"] = original_path
            self.engine.load_model("se_res_unet")


class TestRealSceneInferenceIntegration(unittest.TestCase):
    """Integration tests running inference across all benchmark scenes."""

    @classmethod
    def setUpClass(cls):
        cls.engine = InferenceEngine(default_model="se_res_unet")
        cls.sample_dir = deployment_dir / "sample_inputs"

    def test_all_sample_scenes_execute_successfully(self):
        """Scenes 3, 8, 26, 48, 70 must run inference and return complete valid metadata."""
        sample_ids = [3, 8, 26, 48, 70]

        for s_id in sample_ids:
            tif_path = self.sample_dir / f"{s_id}.tif"
            self.assertTrue(tif_path.exists(), f"Sample scene {s_id}.tif missing")

            results = self.engine.predict_file(tif_path)

            # Dictionary keys contract
            expected_keys = [
                "model_key", "model_name", "threshold", "binary_mask",
                "mask_255", "probabilities", "water_percentage",
                "water_pixels", "land_pixels", "total_pixels",
                "inference_time_ms", "device", "rgb_image", "overlay_image"
            ]
            for k in expected_keys:
                self.assertIn(k, results, f"Missing key '{k}' in inference output")

            # Validate dimensions
            self.assertEqual(results["binary_mask"].shape, (128, 128))
            self.assertEqual(results["mask_255"].shape, (128, 128))
            self.assertEqual(results["rgb_image"].shape, (128, 128, 3))
            self.assertEqual(results["overlay_image"].shape, (128, 128, 3))

            # Validate pixel counts
            self.assertEqual(results["total_pixels"], 16384)
            self.assertEqual(results["water_pixels"] + results["land_pixels"], 16384)

            # Check value ranges
            self.assertTrue(np.all(np.isin(results["mask_255"], [0, 255])))
            self.assertTrue((results["probabilities"] >= 0.0).all())
            self.assertTrue((results["probabilities"] <= 1.0).all())
            self.assertGreater(results["inference_time_ms"], 0.0)

    def test_arid_land_negative_control(self):
        """Scene 70 represents arid land. ResNet50 DeepLabV3+ achieves exactly 0.00% FPR."""
        tif_path = self.sample_dir / "70.tif"
        res_deeplab = self.engine.predict_file(tif_path, model_name="resnet50_deeplabv3p")
        self.assertEqual(res_deeplab["water_percentage"], 0.0, "ResNet50 DeepLabV3+ must achieve 0.00% FPR on arid land")
        self.assertEqual(res_deeplab["water_pixels"], 0)

        # SE-Res-UNet also has extremely low false alarm (< 1.0%)
        res_scratch = self.engine.predict_file(tif_path, model_name="se_res_unet")
        self.assertLess(res_scratch["water_percentage"], 1.0)

    def test_water_scenes_positive_detection(self):
        """Scenes 8 (Lake) and 26 (Coast) must detect significant water bodies (>10%)."""
        res_lake = self.engine.predict_file(self.sample_dir / "8.tif")
        self.assertGreater(res_lake["water_percentage"], 10.0)

        res_coast = self.engine.predict_file(self.sample_dir / "26.tif")
        self.assertGreater(res_coast["water_percentage"], 10.0)

    def test_in_memory_bytesio_inference(self):
        """Inference from raw bytes / BytesIO must yield identical results to disk file reading."""
        tif_path = self.sample_dir / "26.tif"
        with open(tif_path, "rb") as f:
            raw_bytes = f.read()

        res_disk = self.engine.predict_file(tif_path)
        res_bytes = self.engine.predict_file(raw_bytes)
        res_bio = self.engine.predict_file(io.BytesIO(raw_bytes))

        self.assertEqual(res_disk["water_pixels"], res_bytes["water_pixels"])
        self.assertEqual(res_disk["water_pixels"], res_bio["water_pixels"])
        np.testing.assert_array_equal(res_disk["mask_255"], res_bytes["mask_255"])

    def test_threshold_sensitivity_monotonicity(self):
        """Lower threshold (higher recall) must produce >= water pixels than higher threshold (higher precision)."""
        tif_path = self.sample_dir / "26.tif"
        res_loose = self.engine.predict_file(tif_path, threshold=0.10)
        res_strict = self.engine.predict_file(tif_path, threshold=0.85)

        self.assertGreaterEqual(
            res_loose["water_pixels"],
            res_strict["water_pixels"],
            "Lower threshold did not yield greater or equal water pixels!"
        )


class TestEdgeCaseImageryIntegration(unittest.TestCase):
    """Integration tests for abnormal / edge-case inputs."""

    @classmethod
    def setUpClass(cls):
        cls.engine = InferenceEngine(default_model="se_res_unet")

    def test_synthetic_all_zeros_raster(self):
        """Completely black / zeroed raster must run without crash and produce valid output contracts."""
        zeros_arr = np.zeros((128, 128, 12), dtype=np.float32)
        res = self.engine.predict(zeros_arr)
        self.assertEqual(res["total_pixels"], 16384)
        self.assertEqual(res["binary_mask"].shape, (128, 128))
        self.assertFalse(np.isnan(res["probabilities"]).any())
        self.assertFalse(np.isinf(res["probabilities"]).any())

    def test_synthetic_all_ones_raster(self):
        """Uniform saturated raster must be handled safely without NaN or Inf."""
        ones_arr = np.ones((128, 128, 12), dtype=np.float32) * 1000.0
        res = self.engine.predict(ones_arr)
        self.assertFalse(np.isnan(res["probabilities"]).any())
        self.assertFalse(np.isinf(res["probabilities"]).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
