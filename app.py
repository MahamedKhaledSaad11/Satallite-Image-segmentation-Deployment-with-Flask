"""
Flask REST API and Web Application for Multispectral Water Segmentation.
Serves interactive web frontend and provides /predict endpoint for 12-channel GeoTIFF inference.
"""

import os
import io
import base64
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file, Response
from PIL import Image
import numpy as np

from inference import InferenceEngine

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB max upload
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

deployment_dir = Path(__file__).resolve().parent
sample_dir = deployment_dir / "sample_inputs"

# Initialize global inference engine
engine = InferenceEngine(default_model="se_res_unet")


def array_to_base64_png(arr: np.ndarray) -> str:
    """Encodes a uint8 2D (mask) or 3D (RGB) numpy array to a base64 PNG data URL."""
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@app.route("/")
def index():
    """Serves the interactive web UI dashboard."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    """Returns system status, device information, and available models."""
    return jsonify({
        "status": "healthy",
        "service": "Multispectral Satellite Water Segmentation API",
        "device": f"{engine.device} ({engine.device_name})",
        "available_models": {
            k: v["name"] for k, v in engine.MODEL_CONFIGS.items()
        },
        "default_model": engine.default_model_key,
        "supported_formats": [".tif", ".tiff", "12-channel GeoTIFF"]
    })


@app.route("/api/sample_images", methods=["GET"])
def list_sample_images():
    """Returns metadata for built-in sample test images."""
    sample_catalog = [
        {
            "id": "3",
            "filename": "3.tif",
            "name": "Scene 3 - River & Canal",
            "type": "River Network",
            "ground_truth_water": "5.3%",
            "description": "Narrow meandering river canal passing through mixed vegetation."
        },
        {
            "id": "8",
            "filename": "8.tif",
            "name": "Scene 8 - Lake Basin",
            "type": "Open Water Body",
            "ground_truth_water": "15.2%",
            "description": "Medium-sized freshwater lake basin with distinct shoreline."
        },
        {
            "id": "26",
            "filename": "26.tif",
            "name": "Scene 26 - Coastal Wetland",
            "type": "Coastal / Estuary",
            "ground_truth_water": "12.8%",
            "description": "Coastal inlet with high turbidity and shallow sediment."
        },
        {
            "id": "48",
            "filename": "48.tif",
            "name": "Scene 48 - Vegetated Water",
            "type": "Wetland",
            "ground_truth_water": "1.1%",
            "description": "Small pond partially covered by dense aquatic vegetation."
        },
        {
            "id": "70",
            "filename": "70.tif",
            "name": "Scene 70 - Arid Land (Zero Water)",
            "type": "Dry Land / Negative Control",
            "ground_truth_water": "0.0%",
            "description": "Dry mountainous landscape to verify zero false alarm rate."
        }
    ]
    return jsonify({"samples": sample_catalog})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Primary API endpoint for water segmentation inference.
    Accepts:
      - Multipart file upload ('image' or 'file' containing .tif)
      - Query/Form parameters:
          - 'model': 'se_res_unet' | 'resnet50_deeplabv3p' | 'resnet34_unet'
          - 'threshold': float (e.g. 0.3)
          - 'format': 'json' (default) | 'image' | 'overlay'
    """
    file_obj = None
    if "image" in request.files:
        file_obj = request.files["image"]
    elif "file" in request.files:
        file_obj = request.files["file"]

    # Also support sample_id in form or json
    sample_id = request.form.get("sample_id")
    if not file_obj and not sample_id:
        # Check json body
        if request.is_json:
            sample_id = request.json.get("sample_id")

    if not file_obj and not sample_id:
        return jsonify({
            "error": "No image file provided. Upload a .tif file with key 'image' or 'file', or specify 'sample_id'."
        }), 400

    # Validate output format
    output_format = request.form.get("format", request.args.get("format", "json")).lower()
    valid_formats = ["json", "image", "overlay"]
    if output_format not in valid_formats:
        return jsonify({
            "error": f"Invalid format '{output_format}'. Supported formats: {valid_formats}"
        }), 400

    # Validate model parameter if supplied
    model_name = request.form.get("model", request.args.get("model", None))
    if model_name is not None and model_name != "":
        if model_name not in engine.MODEL_CONFIGS:
            return jsonify({
                "error": f"Unknown model architecture '{model_name}'. Available models: {list(engine.MODEL_CONFIGS.keys())}"
            }), 400
    else:
        model_name = None

    # Validate threshold parameter if supplied
    threshold_str = request.form.get("threshold", request.args.get("threshold", None))
    threshold = None
    if threshold_str is not None and threshold_str != "":
        try:
            threshold = float(threshold_str)
            if not (0.0 <= threshold <= 1.0):
                return jsonify({
                    "error": f"Decision threshold must be between 0.0 and 1.0. Received: {threshold}"
                }), 400
        except (ValueError, TypeError):
            return jsonify({
                "error": f"Invalid threshold value '{threshold_str}'. Must be a float between 0.0 and 1.0."
            }), 400

    # Validate file upload if provided
    if file_obj:
        filename = file_obj.filename or "uploaded_image.tif"
        ext = Path(filename).suffix.lower()
        if ext not in [".tif", ".tiff"]:
            return jsonify({
                "error": f"Unsupported file extension '{ext}'. Please upload a 12-channel GeoTIFF image (.tif, .tiff)."
            }), 400

        file_bytes = file_obj.read()
        if len(file_bytes) == 0:
            return jsonify({
                "error": "Uploaded file is empty (0 bytes)."
            }), 400

        try:
            results = engine.predict_file(file_bytes, model_name=model_name, threshold=threshold)
        except ValueError as ve:
            return jsonify({
                "status": "error",
                "message": f"Input validation error: {str(ve)}"
            }), 400
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Inference failed: {str(e)}"
            }), 500
    else:
        sample_file = sample_dir / f"{sample_id}.tif"
        if not sample_file.exists():
            return jsonify({"error": f"Sample image '{sample_id}.tif' not found."}), 404

        try:
            results = engine.predict_file(sample_file, model_name=model_name, threshold=threshold)
            filename = f"sample_{sample_id}.tif"
        except ValueError as ve:
            return jsonify({
                "status": "error",
                "message": f"Input validation error: {str(ve)}"
            }), 400
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Inference failed: {str(e)}"
            }), 500

    # Response formats
    if output_format == "image":
        # Return direct PNG binary mask
        mask_pil = Image.fromarray(results["mask_255"])
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        buf.seek(0)
        return send_file(
            buf,
            mimetype="image/png",
            as_attachment=True,
            download_name=f"{Path(filename).stem}_water_mask.png"
        )
    elif output_format == "overlay":
        overlay_pil = Image.fromarray(results["overlay_image"])
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        buf.seek(0)
        return send_file(
            buf,
            mimetype="image/png",
            as_attachment=True,
            download_name=f"{Path(filename).stem}_water_overlay.png"
        )

    # Default JSON format
    return jsonify({
        "status": "success",
        "filename": filename,
        "model_key": results["model_key"],
        "model_name": results["model_name"],
        "threshold": results["threshold"],
        "water_percentage": results["water_percentage"],
        "water_pixels": results["water_pixels"],
        "land_pixels": results["land_pixels"],
        "total_pixels": results["total_pixels"],
        "inference_time_ms": results["inference_time_ms"],
        "device": results["device"],
        "mask_image": array_to_base64_png(results["mask_255"]),
        "rgb_preview": array_to_base64_png(results["rgb_image"]),
        "overlay_image": array_to_base64_png(results["overlay_image"])
    })


@app.errorhandler(404)
def not_found_handler(e):
    return jsonify({"error": "Resource not found", "status": 404}), 404


@app.errorhandler(405)
def method_not_allowed_handler(e):
    return jsonify({"error": f"HTTP method {request.method} is not allowed on {request.path}.", "status": 405}), 405


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"  Starting Water Segmentation Web App on port {port}...")
    print(f"  Access UI at: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
