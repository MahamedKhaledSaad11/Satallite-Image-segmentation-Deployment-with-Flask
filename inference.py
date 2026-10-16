"""
Production Inference Engine for Multispectral Water Segmentation.
Supports Week 1 Scratch Best (SE-Res-UNet) and Week 2 Pretrained (ResNet50 DeepLabV3+, ResNet34 U-Net).
Handles 12-channel GeoTIFF satellite imagery, domain feature engineering,
GPU-accelerated inference, decision thresholding, and visualization overlay generation.
"""

import sys
import io
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import json
import numpy as np
import tifffile
from PIL import Image
import torch
import torch.nn as nn

deployment_dir = Path(__file__).resolve().parent
project_root = deployment_dir.parent
week1_dir = project_root / "water_segmentation"
part2_dir = project_root / "water_segmentation_part2"

for p in [str(deployment_dir), str(project_root), str(week1_dir), str(part2_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.unet import build_model
from water_segmentation_part2.src.models.pretrained_models import build_model_from_config
from src.data.feature_engineering import build_engineered_features, normalize_engineered_features


class InferenceEngine:
    """Production Inference Engine with multi-model caching and GPU support."""

    MODEL_CONFIGS = {
        "se_res_unet": {
            "checkpoint": deployment_dir / "checkpoints" / "se_res_unet.pt",
            "name": "SE-Res-UNet (Scratch Final - Week 1)",
            "default_threshold": 0.30,
            "type": "scratch"
        },
        "resnet50_deeplabv3p": {
            "checkpoint": deployment_dir / "checkpoints" / "resnet50_deeplabv3p.pt",
            "name": "ResNet-50 + DeepLabV3+ (Pretrained - Week 2)",
            "default_threshold": 0.45,
            "type": "pretrained"
        },
        "resnet34_unet": {
            "checkpoint": deployment_dir / "checkpoints" / "resnet34_unet.pt",
            "name": "ResNet-34 + U-Net (Pretrained - Week 2)",
            "default_threshold": 0.40,
            "type": "pretrained"
        }
    }

    def __init__(self, default_model: str = "se_res_unet", device: Optional[str] = None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        print(f"[InferenceEngine] Initialized on: {self.device} ({self.device_name})")

        # Load normalization statistics
        stats_path = deployment_dir / "metadata" / "engineered_train_stats.json"
        if not stats_path.exists():
            stats_path = week1_dir / "data" / "statistics" / "engineered_train_stats.json"

        with open(stats_path, "r", encoding="utf-8") as f:
            self.engineered_stats = json.load(f)

        # Raw band statistics for NoData replacement
        raw_stats_path = deployment_dir / "metadata" / "train_stats.json"
        if not raw_stats_path.exists():
            raw_stats_path = week1_dir / "data" / "statistics" / "train_stats.json"
        with open(raw_stats_path, "r", encoding="utf-8") as f:
            self.raw_stats = json.load(f)

        # Model cache
        self.models: Dict[str, nn.Module] = {}
        self.thresholds: Dict[str, float] = {}

        # Pre-load default model
        self.load_model(default_model)
        self.default_model_key = default_model

    def load_model(self, model_key: str) -> nn.Module:
        """Loads and caches a model checkpoint into memory."""
        if model_key in self.models:
            return self.models[model_key]

        if model_key not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(self.MODEL_CONFIGS.keys())}")

        cfg = self.MODEL_CONFIGS[model_key]
        ckpt_path = cfg["checkpoint"]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}")

        payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if cfg["type"] == "scratch":
            model = build_model(payload["config"]).to(self.device)
        else:
            model = build_model_from_config(payload["config"]).to(self.device)

        model.load_state_dict(payload["state_dict"])
        model.eval()

        self.models[model_key] = model
        self.thresholds[model_key] = payload.get("optimal_threshold", cfg["default_threshold"])
        print(f"  -> Model '{model_key}' loaded successfully (Default threshold: {self.thresholds[model_key]})")
        return model

    def preprocess_image(self, raw_image: np.ndarray) -> torch.Tensor:
        """
        Preprocesses a raw 12-channel satellite image:
        1. Validates dimensions (H, W, 12).
        2. Fills NoData values in B8 (-9999) with band training mean.
        3. Builds 12 engineered hydrological features (Indices + DEM).
        4. Applies Z-score standardization using training statistics.
        Returns: (1, 12, H, W) float32 PyTorch tensor on target device.
        """
        img = np.asarray(raw_image, dtype=np.float32)

        # Ensure (H, W, C)
        if img.ndim == 3 and img.shape[0] == 12 and img.shape[2] != 12:
            img = np.transpose(img, (1, 2, 0))

        if img.ndim != 3 or img.shape[2] != 12:
            raise ValueError(f"Expected 12-channel multispectral image. Received shape: {img.shape}")

        # Handle NoData (-9999) in B8 (elevation or specific sensor band)
        nodata_mask = (img[:, :, 8] == -9999.0)
        if np.any(nodata_mask):
            fill_val = self.raw_stats["band_means"][8]
            img[:, :, 8][nodata_mask] = fill_val

        # Feature engineering stack (MNDWI, NDWI, AWEI_sh, NDVI, DEM, etc.)
        feat_stack = build_engineered_features(img)

        # Z-score normalization
        norm_chw = normalize_engineered_features(
            feat_img=feat_stack,
            stats=self.engineered_stats,
            method="zscore"
        )

        tensor = torch.from_numpy(norm_chw).unsqueeze(0).float().to(self.device)
        return tensor

    def create_rgb_visualization(self, raw_image: np.ndarray) -> np.ndarray:
        """
        Extracts True-Color RGB channels (Red=B03, Green=B02, Blue=B01)
        and applies 2%-98% percentile contrast stretching for web display.
        Returns: (H, W, 3) uint8 image.
        """
        img = np.asarray(raw_image, dtype=np.float32)
        if img.shape[0] == 12:
            img = np.transpose(img, (1, 2, 0))

        # Sentinel-2 true-color mapping in standard 12-channel stack:
        # B01=Blue (ch 1), B02=Green (ch 2), B03=Red (ch 3)
        r = img[:, :, 3]
        g = img[:, :, 2]
        b = img[:, :, 1]
        rgb = np.stack([r, g, b], axis=-1)

        # Robust percentile contrast stretching per channel
        stretched = np.zeros_like(rgb, dtype=np.uint8)
        for c in range(3):
            band = rgb[:, :, c]
            p2 = np.percentile(band, 2)
            p98 = np.percentile(band, 98)
            if p98 > p2:
                norm = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
            else:
                norm = np.zeros_like(band)
            stretched[:, :, c] = (norm * 255).astype(np.uint8)

        return stretched

    def create_overlay_visualization(self, rgb_image: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Creates an RGBA or blended RGB image with vibrant Cyan/Azure water overlay on satellite RGB.
        Returns: (H, W, 3) uint8 image.
        """
        overlay = rgb_image.copy()
        # Vibrant cyan/azure water color: RGB(0, 180, 255)
        water_color = np.array([0, 180, 255], dtype=np.float32)

        is_water = binary_mask > 0
        if np.any(is_water):
            overlay[is_water] = (
                (1.0 - alpha) * rgb_image[is_water].astype(np.float32) + alpha * water_color
            ).astype(np.uint8)

        return overlay

    def predict(
        self,
        raw_image: np.ndarray,
        model_name: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Runs complete inference pipeline on a 12-channel satellite image.
        Returns dictionary with binary mask, metrics, probabilities, and visualizations.
        """
        model_key = model_name if model_name else self.default_model_key
        if model_key not in self.models:
            self.load_model(model_key)

        model = self.models[model_key]
        decision_threshold = float(threshold if threshold is not None else self.thresholds[model_key])

        start_time = time.perf_counter()

        # 1. Preprocess
        tensor = self.preprocess_image(raw_image)

        # 2. Forward pass
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Decision thresholding
        binary_mask = (probs > decision_threshold).astype(np.uint8)

        # 4. Compute hydrological metrics
        total_pixels = int(binary_mask.size)
        water_pixels = int(np.sum(binary_mask > 0))
        land_pixels = total_pixels - water_pixels
        water_percentage = float((water_pixels / total_pixels) * 100.0)

        # 5. Visualizations
        rgb_image = self.create_rgb_visualization(raw_image)
        overlay_image = self.create_overlay_visualization(rgb_image, binary_mask)

        return {
            "model_key": model_key,
            "model_name": self.MODEL_CONFIGS[model_key]["name"],
            "threshold": decision_threshold,
            "binary_mask": binary_mask,             # (H, W) {0, 1}
            "mask_255": binary_mask * 255,          # (H, W) {0, 255}
            "probabilities": probs,                 # (H, W) [0.0, 1.0]
            "water_percentage": round(water_percentage, 2),
            "water_pixels": water_pixels,
            "land_pixels": land_pixels,
            "total_pixels": total_pixels,
            "inference_time_ms": round(inference_time_ms, 2),
            "device": str(self.device),
            "rgb_image": rgb_image,
            "overlay_image": overlay_image
        }

    def predict_file(
        self,
        file_input: Union[str, Path, io.BytesIO, bytes],
        model_name: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Reads a GeoTIFF from file path, bytes, or BytesIO buffer and predicts water mask."""
        if isinstance(file_input, (str, Path)):
            raw_img = tifffile.imread(str(file_input))
        elif isinstance(file_input, bytes):
            with io.BytesIO(file_input) as bio:
                raw_img = tifffile.imread(bio)
        else:
            raw_img = tifffile.imread(file_input)

        return self.predict(raw_img, model_name=model_name, threshold=threshold)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Water Segmentation Inference Engine (CLI)")
    parser.add_argument("--image", type=str, required=True, help="Path to input 12-channel .tif image")
    parser.add_argument("--output", type=str, default="predicted_water_mask.png", help="Path to save output binary mask")
    parser.add_argument("--model", type=str, default="se_res_unet", choices=["se_res_unet", "resnet50_deeplabv3p", "resnet34_unet"])
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda or cpu)")
    parser.add_argument("--save_overlay", action="store_true", help="Also save RGB overlay visualization")
    args = parser.parse_args()

    engine = InferenceEngine(default_model=args.model, device=args.device)
    print(f"Running inference on: {args.image}")

    results = engine.predict_file(args.image, model_name=args.model, threshold=args.threshold)

    # Save mask
    mask_img = Image.fromarray(results["mask_255"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(str(output_path))
    print(f"Binary mask saved to: {output_path}")

    if args.save_overlay:
        overlay_path = output_path.with_name(f"{output_path.stem}_overlay.png")
        Image.fromarray(results["overlay_image"]).save(str(overlay_path))
        print(f"Overlay visualization saved to: {overlay_path}")

    print("\n" + "=" * 50)
    print(f"Model:              {results['model_name']}")
    print(f"Decision Threshold: {results['threshold']}")
    print(f"Water Coverage:     {results['water_percentage']}% ({results['water_pixels']:,} / {results['total_pixels']:,} pixels)")
    print(f"Inference Time:     {results['inference_time_ms']} ms")
    print(f"Compute Device:     {results['device']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
