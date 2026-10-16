"""
Unit Tests for AquaSeg Multispectral Water Segmentation.
Validates isolated components:
1. Hydrological remote sensing formulas (MNDWI, NDWI, NDVI, AWEI_sh, safe_normalized_diff)
2. Image preprocessing, NoData replacement, shape validation, and tensor transformations
3. Postprocessing, decision thresholding, metrics calculation, and base64 PNG serialization
4. Neural network architectures (SE-Res-UNet, DeepLabV3+, U-Net) forward pass shapes and contracts
"""

import sys
import unittest
import io
import base64
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# Add project paths
deployment_dir = Path(__file__).resolve().parent.parent
project_root = deployment_dir.parent
for p in [str(deployment_dir), str(project_root), str(project_root / "water_segmentation"), str(project_root / "water_segmentation_part2")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data.feature_engineering import (
    safe_normalized_diff,
    compute_mndwi,
    compute_ndwi,
    compute_ndvi,
    compute_awei_sh
)
from src.models.unet import build_model
from water_segmentation_part2.src.models.pretrained_models import build_model_from_config
from inference import InferenceEngine
from app import array_to_base64_png


class TestHydrologicalFormulas(unittest.TestCase):
    """Unit tests for remote sensing hydrological index formulas."""

    def test_safe_normalized_diff_regular(self):
        a = np.array([0.4, 0.6], dtype=np.float32)
        b = np.array([0.2, 0.2], dtype=np.float32)
        # (0.4 - 0.2) / (0.4 + 0.2) = 0.2 / 0.6 = 0.3333
        # (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5000
        res = safe_normalized_diff(a, b)
        np.testing.assert_allclose(res, [0.33333334, 0.5], rtol=1e-5)

    def test_safe_normalized_diff_zero_denominator(self):
        """Denominator near zero must be clamped safely without NaN or Inf."""
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        res = safe_normalized_diff(a, b)
        self.assertFalse(np.isnan(res).any())
        self.assertFalse(np.isinf(res).any())
        self.assertTrue(np.all(np.abs(res) <= 1.0))

    def test_safe_normalized_diff_clamping(self):
        """Indices must strictly remain in [-1.0, 1.0]."""
        a = np.array([10.0, -5.0], dtype=np.float32)
        b = np.array([0.0, 5.0], dtype=np.float32)
        res = safe_normalized_diff(a, b)
        self.assertTrue((res >= -1.0).all())
        self.assertTrue((res <= 1.0).all())

    def test_compute_mndwi_water_vs_soil(self):
        """Water has high Green and very low SWIR1 -> Positive MNDWI. Soil has high SWIR1 -> Negative MNDWI."""
        # Simulated water pixel
        green_water = np.array([[0.35]], dtype=np.float32)
        swir1_water = np.array([[0.05]], dtype=np.float32)
        mndwi_water = compute_mndwi(green_water, swir1_water)
        self.assertGreater(mndwi_water[0, 0], 0.5)

        # Simulated soil/dry land pixel
        green_soil = np.array([[0.10]], dtype=np.float32)
        swir1_soil = np.array([[0.40]], dtype=np.float32)
        mndwi_soil = compute_mndwi(green_soil, swir1_soil)
        self.assertLess(mndwi_soil[0, 0], -0.4)

    def test_compute_ndwi(self):
        """NDWI = (Green - NIR) / (Green + NIR)."""
        green = np.array([0.3], dtype=np.float32)
        nir = np.array([0.1], dtype=np.float32)
        expected = (0.3 - 0.1) / (0.3 + 0.1)  # 0.2 / 0.4 = 0.5
        np.testing.assert_allclose(compute_ndwi(green, nir), [expected], rtol=1e-5)

    def test_compute_ndvi(self):
        """NDVI = (NIR - Red) / (NIR + Red)."""
        nir = np.array([0.5], dtype=np.float32)
        red = np.array([0.1], dtype=np.float32)
        expected = (0.5 - 0.1) / (0.5 + 0.1)  # 0.4 / 0.6 = 0.6667
        np.testing.assert_allclose(compute_ndvi(nir, red), [expected], rtol=1e-5)

    def test_compute_awei_sh(self):
        """AWEI_sh formula: Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2."""
        blue = np.array([0.15], dtype=np.float32)
        green = np.array([0.20], dtype=np.float32)
        nir = np.array([0.05], dtype=np.float32)
        swir1 = np.array([0.02], dtype=np.float32)
        swir2 = np.array([0.01], dtype=np.float32)

        expected = 0.15 + (2.5 * 0.20) - (1.5 * (0.05 + 0.02)) - (0.25 * 0.01)
        res = compute_awei_sh(blue, green, nir, swir1, swir2)
        np.testing.assert_allclose(res, [expected], rtol=1e-5)


class TestPreprocessingUnit(unittest.TestCase):
    """Unit tests for raster preprocessing and validation."""

    @classmethod
    def setUpClass(cls):
        cls.engine = InferenceEngine(default_model="se_res_unet", device="cpu")

    def test_preprocess_transposition(self):
        """Images given as (12, H, W) must be correctly transposed to (H, W, 12) then returned as (1, 12, H, W)."""
        arr_ch_first = np.zeros((12, 64, 64), dtype=np.float32)
        tensor = self.engine.preprocess_image(arr_ch_first)
        self.assertEqual(tensor.shape, (1, 12, 64, 64))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_preprocess_invalid_channel_count(self):
        """Arrays with incorrect channel count (e.g. 3 channels) must raise ValueError."""
        rgb_3ch = np.zeros((128, 128, 3), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            self.engine.preprocess_image(rgb_3ch)
        self.assertIn("Expected 12-channel multispectral image", str(ctx.exception))

    def test_preprocess_invalid_dimensions_2d(self):
        """2D array must raise ValueError."""
        arr_2d = np.zeros((128, 128), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.engine.preprocess_image(arr_2d)

    def test_preprocess_nodata_replacement(self):
        """NoData values (-9999.0) in band index 8 must be replaced by band mean."""
        raw = np.ones((32, 32, 12), dtype=np.float32) * 500.0
        raw[5, 5, 8] = -9999.0
        tensor = self.engine.preprocess_image(raw)
        # Should not have extreme negative values
        self.assertFalse(torch.isnan(tensor).any())
        self.assertFalse(torch.isinf(tensor).any())
        self.assertGreater(tensor[0, 8, 5, 5].item(), -50.0)


class TestPostprocessingUnit(unittest.TestCase):
    """Unit tests for decision thresholding, metrics, and base64 serialization."""

    def test_thresholding_logic(self):
        """Values >= threshold must be 255 (Water), < threshold must be 0 (Land)."""
        probs = np.array([
            [0.10, 0.29],
            [0.30, 0.95]
        ], dtype=np.float32)

        thresh = 0.30
        mask_binary = (probs >= thresh).astype(np.uint8)
        mask_255 = mask_binary * 255

        expected_binary = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        expected_255 = np.array([[0, 0], [255, 255]], dtype=np.uint8)

        np.testing.assert_array_equal(mask_binary, expected_binary)
        np.testing.assert_array_equal(mask_255, expected_255)

    def test_water_percentage_calculation(self):
        """Water percentage calculation."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:25, :] = 1  # 2500 water pixels out of 10000 = 25.0%

        water_px = int(np.sum(mask))
        total_px = mask.size
        pct = (water_px / total_px) * 100.0

        self.assertEqual(water_px, 2500)
        self.assertEqual(total_px, 10000)
        self.assertEqual(pct, 25.0)

    def test_array_to_base64_png(self):
        """Serialization of uint8 numpy array to base64 PNG data URL."""
        test_img = (np.eye(64, dtype=np.uint8) * 255)
        data_url = array_to_base64_png(test_img)

        self.assertTrue(data_url.startswith("data:image/png;base64,"))

        # Decode and verify it is a valid PNG
        raw_b64 = data_url.split(",")[1]
        decoded_bytes = base64.b64decode(raw_b64)
        pil_img = Image.open(io.BytesIO(decoded_bytes))

        self.assertEqual(pil_img.format, "PNG")
        self.assertEqual(pil_img.size, (64, 64))

    def test_color_overlay_generation(self):
        """Overlay must produce uint8 RGB array of matching spatial dimensions."""
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1

        engine = InferenceEngine(default_model="se_res_unet", device="cpu")
        overlay = engine.create_overlay_visualization(rgb, mask)

        self.assertEqual(overlay.shape, (64, 64, 3))
        self.assertEqual(overlay.dtype, np.uint8)
        # Marked pixels must have cyan tint
        self.assertGreater(overlay[20, 20, 2], 0)  # Blue channel high in cyan


class TestModelArchitectures(unittest.TestCase):
    """Unit tests for neural network architectures."""

    def test_se_res_unet_forward_shape(self):
        """SE-Res-UNet forward pass contract: (B, 12, H, W) -> (B, 1, H, W)."""
        config = {
            "name": "se_res_unet",
            "in_channels": 12,
            "base_channels": 16,
            "channel_mults": [1, 2, 4],
            "attention": "se",
            "residual": True,
            "dropout": 0.0
        }
        model = build_model(config)
        model.eval()

        dummy_x = torch.randn(2, 12, 64, 64)
        with torch.no_grad():
            logits = model(dummy_x)

        self.assertEqual(logits.shape, (2, 1, 64, 64))
        self.assertFalse(torch.isnan(logits).any())

    def test_pretrained_deeplabv3p_forward_shape(self):
        """DeepLabV3Plus forward pass contract: (B, 12, H, W) -> (B, 1, H, W)."""
        config = {
            "name": "deeplabv3plus",
            "encoder_name": "resnet50",
            "in_channels": 12,
            "classes": 1,
            "encoder_weights": None
        }
        model = build_model_from_config(config)
        model.eval()

        dummy_x = torch.randn(1, 12, 128, 128)
        with torch.no_grad():
            logits = model(dummy_x)

        self.assertEqual(logits.shape, (1, 1, 128, 128))
        self.assertFalse(torch.isnan(logits).any())

    def test_pretrained_unet_forward_shape(self):
        """U-Net with ResNet34 encoder forward pass contract: (B, 12, H, W) -> (B, 1, H, W)."""
        config = {
            "name": "unet",
            "encoder_name": "resnet34",
            "in_channels": 12,
            "classes": 1,
            "encoder_weights": None
        }
        model = build_model_from_config(config)
        model.eval()

        dummy_x = torch.randn(1, 12, 128, 128)
        with torch.no_grad():
            logits = model(dummy_x)

        self.assertEqual(logits.shape, (1, 1, 128, 128))
        self.assertFalse(torch.isnan(logits).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
