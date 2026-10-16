# Model Checkpoints Directory

Due to GitHub's file size limit (100 MB maximum per file), trained `.pt` weight files are excluded from git tracking.

## Generating Checkpoints Locally
You can generate the clean, lightweight inference checkpoints at any time by running:

```powershell
python export_weights.py
```

This will automatically create:
- `checkpoints/se_res_unet.pt` (30.5 MB)
- `checkpoints/resnet50_deeplabv3p.pt` (102.2 MB)
- `checkpoints/resnet34_unet.pt` (93.5 MB)
