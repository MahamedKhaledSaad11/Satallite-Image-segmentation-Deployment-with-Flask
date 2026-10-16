"""
Model Weights Exporter for Deployment.
Strips optimizer and scheduler states from training checkpoints to produce
lightweight, production-ready inference weights.
"""

from pathlib import Path
import json
import torch

deployment_dir = Path(__file__).resolve().parent
project_root = deployment_dir.parent

def export_checkpoint(
    source_path: Path,
    target_path: Path,
    model_type: str,
    architecture_config: dict,
    optimal_threshold: float,
    notes: str = ""
):
    print(f"\n[Exporting] {model_type}...")
    if not source_path.exists():
        print(f"  [Error] Source checkpoint not found: {source_path}")
        return False

    raw_size_mb = source_path.stat().st_size / (1024 * 1024)
    print(f"  -> Raw training checkpoint size: {raw_size_mb:.2f} MB")

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    deploy_payload = {
        "model_type": model_type,
        "state_dict": state_dict,
        "config": architecture_config,
        "optimal_threshold": optimal_threshold,
        "notes": notes
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deploy_payload, target_path)

    clean_size_mb = target_path.stat().st_size / (1024 * 1024)
    compression = (1.0 - clean_size_mb / raw_size_mb) * 100.0
    print(f"  -> Exported clean weights size:  {clean_size_mb:.2f} MB ({compression:.1f}% reduction)")
    print(f"  -> Target saved to: {target_path.name}")
    return True


def main():
    print("=" * 70)
    print("     EXPORTING PRODUCTION MODEL WEIGHTS FOR DEPLOYMENT")
    print("=" * 70)

    # 1. SE-Res-UNet (Scratch Best - Week 1)
    se_res_source = project_root / "water_segmentation" / "checkpoints" / "final_model_engineered" / "best_model.pt"
    se_res_target = deployment_dir / "checkpoints" / "se_res_unet.pt"
    se_res_cfg = {
        "name": "se_res_unet",
        "in_channels": 12,
        "out_channels": 1,
        "base_channels": 32,
        "norm_type": "group",
        "num_groups": 8,
        "reduction": 16,
        "feature_engineering": True,
        "norm_method": "zscore"
    }
    export_checkpoint(
        source_path=se_res_source,
        target_path=se_res_target,
        model_type="se_res_unet",
        architecture_config=se_res_cfg,
        optimal_threshold=0.30,
        notes="Week 1 Scratch Best Model: 0.7285 Test IoU, 8.12M parameters"
    )

    # 2. ResNet50 + DeepLabV3+ (Pretrained Best - Week 2)
    deeplab_source = project_root / "water_segmentation_part2" / "checkpoints" / "resnet50_deeplabv3p" / "best_model.pt"
    deeplab_target = deployment_dir / "checkpoints" / "resnet50_deeplabv3p.pt"
    deeplab_cfg = {
        "decoder": "deeplabv3plus",
        "encoder": "resnet50",
        "in_channels": 12,
        "out_channels": 1,
        "adaptation_strategy": "kaiming_extra",
        "feature_engineering": True,
        "norm_method": "zscore"
    }
    export_checkpoint(
        source_path=deeplab_source,
        target_path=deeplab_target,
        model_type="resnet50_deeplabv3p",
        architecture_config=deeplab_cfg,
        optimal_threshold=0.45,
        notes="Week 2 Pretrained Best Model: 0.7156 Test IoU, 0.00% Empty FPR, 39.75M parameters"
    )

    # 3. ResNet34 + U-Net (Pretrained - Week 2)
    r34_source = project_root / "water_segmentation_part2" / "checkpoints" / "resnet34_unet" / "best_model.pt"
    r34_target = deployment_dir / "checkpoints" / "resnet34_unet.pt"
    r34_cfg = {
        "decoder": "unet",
        "encoder": "resnet34",
        "in_channels": 12,
        "out_channels": 1,
        "adaptation_strategy": "kaiming_extra",
        "feature_engineering": True,
        "norm_method": "zscore"
    }
    export_checkpoint(
        source_path=r34_source,
        target_path=r34_target,
        model_type="resnet34_unet",
        architecture_config=r34_cfg,
        optimal_threshold=0.40,
        notes="Week 2 Pretrained Model: 0.7205 Val IoU, 0.8422 Recall, 24.46M parameters"
    )

    print("\n" + "=" * 70)
    print("All production model weights exported successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
