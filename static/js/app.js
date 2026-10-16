/**
 * AquaSeg — Modern Remote Sensing Workstation Script
 * Orchestrates satellite inference, multi-stage pipeline animations,
 * live pixel HUD inspection, interactive split-slider, and telemetry counter animations.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements: Inputs
    const dropArea = document.getElementById("drop-area");
    const fileInput = document.getElementById("file-input");
    const fileChip = document.getElementById("file-chip");
    const fileChipName = document.getElementById("file-chip-name");
    const fileChipRemove = document.getElementById("file-chip-remove");
    const btnRun = document.getElementById("btn-run");
    const trayChips = document.querySelectorAll(".tray-chip");

    const modelSelect = document.getElementById("model-select");
    const modelHelp = document.getElementById("model-help");
    const thresholdSlider = document.getElementById("threshold-slider");
    const thresholdVal = document.getElementById("threshold-val");
    const resetThreshBtn = document.getElementById("reset-thresh-btn");

    // DOM Elements: States & Stepper
    const slateEmpty = document.getElementById("slate-empty");
    const slateBusy = document.getElementById("slate-busy");
    const busySub = document.getElementById("busy-sub");
    const dashboard = document.getElementById("dashboard");
    const pipeSteps = [
        document.getElementById("pipe-step-1"),
        document.getElementById("pipe-step-2"),
        document.getElementById("pipe-step-3"),
        document.getElementById("pipe-step-4")
    ];

    // DOM Elements: Telemetry
    const statPct = document.getElementById("stat-pct");
    const statWaterType = document.getElementById("stat-water-type");
    const statBar = document.getElementById("stat-bar");
    const statWaterPx = document.getElementById("stat-water-px");
    const statLandPx = document.getElementById("stat-land-px");
    const statLatency = document.getElementById("stat-latency");
    const statDevice = document.getElementById("stat-device");
    const statModel = document.getElementById("stat-model");
    const statThreshold = document.getElementById("stat-threshold");
    const maskPixelCount = document.getElementById("mask-pixel-count");
    const overlayCovTag = document.getElementById("overlay-cov-tag");

    // DOM Elements: Visuals & Zoom Boxes
    const imgRgb = document.getElementById("img-rgb");
    const imgMask = document.getElementById("img-mask");
    const imgOverlay = document.getElementById("img-overlay");
    const splitBg = document.getElementById("split-bg");
    const splitFg = document.getElementById("split-fg");
    const splitClip = document.getElementById("split-clip");
    const splitHandle = document.getElementById("split-handle");
    const splitBox = document.getElementById("split-box");

    // Pixel HUDs
    const hudRgb = document.getElementById("hud-rgb");
    const hudMask = document.getElementById("hud-mask");
    const hudOverlay = document.getElementById("hud-overlay");

    // Tint Dots
    const tintDots = document.querySelectorAll(".tint-dot");

    // Views & Export
    const btnViewGrid = document.getElementById("btn-view-grid");
    const btnViewSlider = document.getElementById("btn-view-slider");
    const viewGallery = document.getElementById("view-gallery");
    const viewSplit = document.getElementById("view-split");
    const btnDownload = document.getElementById("btn-download");

    // Sensor Specs Modal
    const btnSensorModal = document.getElementById("btn-sensor-modal");
    const modalSensor = document.getElementById("modal-sensor");
    const btnModalClose = document.getElementById("btn-modal-close");
    const btnModalGotit = document.getElementById("btn-modal-gotit");

    // Toast Container
    const toastContainer = document.getElementById("toast-container");

    // Offscreen Canvas for Real-Time Pixel Sampling
    const offCanvas = document.createElement("canvas");
    offCanvas.width = 128;
    offCanvas.height = 128;
    const offCtx = offCanvas.getContext("2d", { willReadFrequently: true });
    let maskPixelData = null;

    // Model Architecture Metadata
    const MODEL_SPECS = {
        "se_res_unet": {
            optimal_thresh: 0.30,
            help: "Recommended. Lightweight scratch network (8.1M params) with Squeeze-and-Excitation channel attention. Highest test IoU (0.7285)."
        },
        "resnet50_deeplabv3p": {
            optimal_thresh: 0.45,
            help: "Zero False Positives (0.0% FPR). ResNet-50 backbone with Atrous Spatial Pyramid Pooling (ASPP). Ideal for dry-land rejection."
        },
        "resnet34_unet": {
            optimal_thresh: 0.40,
            help: "High Sensitivity (84.2% Recall). ImageNet pretrained encoder with fine-grained U-Net skip connections."
        }
    };

    let activeFile = null;
    let activeSampleId = null;
    let lastResultData = null;

    // 2. Model Selection
    modelSelect.addEventListener("change", () => {
        const spec = MODEL_SPECS[modelSelect.value];
        if (spec) {
            modelHelp.textContent = spec.help;
            thresholdSlider.value = spec.optimal_thresh;
            thresholdVal.textContent = spec.optimal_thresh.toFixed(2);
            resetThreshBtn.textContent = `Reset (${spec.optimal_thresh.toFixed(2)})`;
            showToast(`Switched architecture: ${modelSelect.options[modelSelect.selectedIndex].text.split('•')[0].trim()}`);
        }
    });

    // 3. Threshold Slider
    thresholdSlider.addEventListener("input", () => {
        thresholdVal.textContent = parseFloat(thresholdSlider.value).toFixed(2);
    });

    resetThreshBtn.addEventListener("click", () => {
        const spec = MODEL_SPECS[modelSelect.value];
        if (spec) {
            thresholdSlider.value = spec.optimal_thresh;
            thresholdVal.textContent = spec.optimal_thresh.toFixed(2);
            showToast(`Threshold reset to τ = ${spec.optimal_thresh.toFixed(2)}`);
        }
    });

    // 4. File Drop & Input
    dropArea.addEventListener("click", (e) => {
        if (e.target !== fileChipRemove) {
            fileInput.click();
        }
    });

    dropArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropArea.classList.add("dragover");
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.classList.remove("dragover");
    });

    dropArea.addEventListener("drop", (e) => {
        e.preventDefault();
        dropArea.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    fileChipRemove.addEventListener("click", (e) => {
        e.stopPropagation();
        activeFile = null;
        fileInput.value = "";
        fileChip.style.display = "none";
        dropArea.querySelector(".drop-placeholder").style.display = "block";
        btnRun.disabled = true;
    });

    function handleFileSelect(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (ext !== "tif" && ext !== "tiff") {
            showToast("Please choose a 12-channel multispectral GeoTIFF image (.tif, .tiff)", "error");
            return;
        }

        activeFile = file;
        activeSampleId = null;
        fileChipName.textContent = file.name;
        dropArea.querySelector(".drop-placeholder").style.display = "none";
        fileChip.style.display = "flex";
        btnRun.disabled = false;
        clearTrayActiveState();
        showToast(`Loaded GeoTIFF: ${file.name}`);
    }

    // 5. Sample Tray Chips (1-Click Run)
    trayChips.forEach((chip, idx) => {
        chip.addEventListener("click", () => {
            activeSampleId = chip.getAttribute("data-id");
            activeFile = null;
            fileInput.value = "";
            fileChip.style.display = "none";
            dropArea.querySelector(".drop-placeholder").style.display = "block";
            btnRun.disabled = false;

            clearTrayActiveState();
            chip.classList.add("active");

            // Execute immediately for instant feedback
            executeInference();
        });
    });

    function clearTrayActiveState() {
        trayChips.forEach(c => c.classList.remove("active"));
    }

    // 6. Action Button
    btnRun.addEventListener("click", () => {
        executeInference();
    });

    // Keyboard Shortcuts
    document.addEventListener("keydown", (e) => {
        if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;

        if (e.code === "Space") {
            e.preventDefault();
            if (!btnRun.disabled) {
                executeInference();
            }
        } else if (e.code === "KeyR") {
            resetThreshBtn.click();
        } else if (["Digit1", "Digit2", "Digit3", "Digit4", "Digit5"].includes(e.code)) {
            const index = parseInt(e.code.replace("Digit", ""), 10) - 1;
            if (trayChips[index]) {
                trayChips[index].click();
            }
        }
    });

    // 7. Inference Execution & Animated Pipeline Stepper
    let stepperInterval = null;

    function resetStepper() {
        if (stepperInterval) clearInterval(stepperInterval);
        pipeSteps.forEach(step => {
            step.className = "pipe-step pending";
        });
    }

    function runStepperAnimation() {
        resetStepper();
        let currentStep = 0;
        pipeSteps[0].className = "pipe-step active";

        stepperInterval = setInterval(() => {
            if (currentStep < pipeSteps.length - 1) {
                pipeSteps[currentStep].className = "pipe-step completed";
                currentStep++;
                pipeSteps[currentStep].className = "pipe-step active";
            }
        }, 120);
    }

    function completeStepper() {
        if (stepperInterval) clearInterval(stepperInterval);
        pipeSteps.forEach(step => {
            step.className = "pipe-step completed";
        });
    }

    function executeInference() {
        if (!activeFile && !activeSampleId) {
            showToast("Please select a sample scene or upload a .tif satellite tile.", "error");
            return;
        }

        const formData = new FormData();
        formData.append("model", modelSelect.value);
        formData.append("threshold", thresholdSlider.value);
        formData.append("format", "json");

        if (activeFile) {
            formData.append("image", activeFile);
            busySub.textContent = `Streaming ${activeFile.name} to GPU`;
        } else if (activeSampleId) {
            formData.append("sample_id", activeSampleId);
            busySub.textContent = `Extracting Scene ${activeSampleId} • 12 Multispectral Bands`;
        }

        // Show Busy State & Trigger Animated Stepper
        slateEmpty.style.display = "none";
        dashboard.style.display = "none";
        slateBusy.style.display = "flex";
        btnRun.disabled = true;
        runStepperAnimation();

        const startTime = performance.now();

        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(async res => {
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.error || data.message || "Segmentation request failed");
            }
            return data;
        })
        .then(data => {
            completeStepper();
            lastResultData = data;
            // Short 100ms graceful pause so the user sees all checkmarks complete
            setTimeout(() => {
                renderResults(data);
            }, 100);
        })
        .catch(err => {
            showToast(`Inference Error: ${err.message}`, "error");
            slateEmpty.style.display = "flex";
        })
        .finally(() => {
            setTimeout(() => {
                slateBusy.style.display = "none";
                btnRun.disabled = false;
            }, 100);
        });
    }

    // 8. Render Results & Animate Telemetry
    function renderResults(data) {
        // Populate Visuals
        imgRgb.src = data.rgb_preview;
        imgMask.src = data.mask_image;
        imgOverlay.src = data.overlay_image;

        // Split Slider layers
        splitBg.src = data.rgb_preview;
        splitFg.src = data.overlay_image;

        // Export button setup
        btnDownload.href = data.mask_image;
        btnDownload.download = `${data.filename ? data.filename.split('.')[0] : 'water'}_mask.png`;

        // Cache binary mask in offscreen canvas for Real-Time Pixel HUD sampling
        const tempImg = new Image();
        tempImg.crossOrigin = "anonymous";
        tempImg.onload = () => {
            offCtx.drawImage(tempImg, 0, 0, 128, 128);
            try {
                maskPixelData = offCtx.getImageData(0, 0, 128, 128).data;
            } catch (e) {
                maskPixelData = null;
            }
        };
        tempImg.src = data.mask_image;

        // Telemetry: Coverage count-up animation
        animateCounter(statPct, data.water_percentage, "%");
        statBar.style.width = `${Math.min(data.water_percentage, 100)}%`;

        // Hydrological classification tag
        const cov = data.water_percentage;
        if (cov === 0) {
            statWaterType.textContent = "Dry Terrain";
            statWaterType.style.background = "rgba(245, 158, 11, 0.15)";
            statWaterType.style.color = "#f59e0b";
        } else if (cov < 5) {
            statWaterType.textContent = "Canal / Stream";
            statWaterType.style.background = "rgba(0, 229, 255, 0.15)";
            statWaterType.style.color = "var(--cyan-primary)";
        } else if (cov < 25) {
            statWaterType.textContent = "Water Body";
            statWaterType.style.background = "rgba(0, 229, 255, 0.2)";
            statWaterType.style.color = "var(--cyan-primary)";
        } else {
            statWaterType.textContent = "Open Basin";
            statWaterType.style.background = "rgba(16, 185, 129, 0.2)";
            statWaterType.style.color = "#34d399";
        }

        statWaterPx.textContent = data.water_pixels.toLocaleString();
        statLandPx.textContent = `Land: ${data.land_pixels.toLocaleString()} px`;
        statLatency.textContent = data.inference_time_ms.toFixed(1);
        statDevice.textContent = data.device ? data.device.split('with')[0].trim() : "CUDA GPU";
        statModel.textContent = data.model_name.split('(')[0].trim();
        statThreshold.textContent = `Threshold: τ = ${data.threshold.toFixed(2)}`;

        maskPixelCount.textContent = `${data.water_pixels.toLocaleString()} water px`;
        overlayCovTag.textContent = `${cov.toFixed(2)}% Water`;

        dashboard.style.display = "flex";
    }

    // Number counter animation helper with ease-out cubic
    function animateCounter(element, targetVal, suffix = "") {
        const duration = 650;
        const start = performance.now();
        const startVal = 0;

        function update(time) {
            const progress = Math.min((time - start) / duration, 1);
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = startVal + (targetVal - startVal) * easeOut;
            element.textContent = `${current.toFixed(2)}${suffix}`;

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.textContent = `${targetVal.toFixed(2)}${suffix}`;
            }
        }
        requestAnimationFrame(update);
    }

    // 9. Interactive Pixel HUD Inspector (Hover over images)
    function setupPixelInspector(boxElem, hudElem) {
        boxElem.addEventListener("mousemove", (e) => {
            const rect = boxElem.getBoundingClientRect();
            const relX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
            const relY = Math.max(0, Math.min(e.clientY - rect.top, rect.height));

            const pixelX = Math.floor((relX / rect.width) * 128);
            const pixelY = Math.floor((relY / rect.height) * 128);

            let classTag = "";
            if (maskPixelData) {
                const idx = (pixelY * 128 + pixelX) * 4;
                const isWater = maskPixelData[idx] > 120;
                classTag = isWater ? " • Water (Water Body)" : " • Land (Dry Surface)";
            }

            hudElem.textContent = `[X: ${pixelX}, Y: ${pixelY}]${classTag}`;
        });
    }

    setupPixelInspector(document.querySelector('[data-target="rgb"]'), hudRgb);
    setupPixelInspector(document.querySelector('[data-target="mask"]'), hudMask);
    setupPixelInspector(document.querySelector('[data-target="overlay"]'), hudOverlay);

    // 10. Overlay Tint Selector
    tintDots.forEach(dot => {
        dot.addEventListener("click", () => {
            tintDots.forEach(d => d.classList.remove("active"));
            dot.classList.add("active");
            const color = dot.getAttribute("data-color");

            if (color === "cyan") {
                imgOverlay.style.filter = "none";
                splitFg.style.filter = "none";
            } else if (color === "emerald") {
                imgOverlay.style.filter = "hue-rotate(55deg) saturate(1.3)";
                splitFg.style.filter = "hue-rotate(55deg) saturate(1.3)";
            } else if (color === "amber") {
                imgOverlay.style.filter = "hue-rotate(185deg) saturate(1.6)";
                splitFg.style.filter = "hue-rotate(185deg) saturate(1.6)";
            }
            showToast(`Overlay tint set to ${color}`);
        });
    });

    // 11. View Switcher (Gallery vs Split View)
    btnViewGrid.addEventListener("click", () => {
        btnViewGrid.classList.add("active");
        btnViewSlider.classList.remove("active");
        viewGallery.style.display = "grid";
        viewSplit.style.display = "none";
    });

    btnViewSlider.addEventListener("click", () => {
        btnViewSlider.classList.add("active");
        btnViewGrid.classList.remove("active");
        viewGallery.style.display = "none";
        viewSplit.style.display = "flex";
    });

    // 12. Interactive Split Comparison Slider Logic
    let isDraggingSplit = false;

    splitBox.addEventListener("mousedown", (e) => {
        isDraggingSplit = true;
        updateSplitHandle(e);
    });

    window.addEventListener("mouseup", () => {
        isDraggingSplit = false;
    });

    window.addEventListener("mousemove", (e) => {
        if (isDraggingSplit) {
            updateSplitHandle(e);
        }
    });

    // Touch support for split slider
    splitBox.addEventListener("touchstart", (e) => {
        isDraggingSplit = true;
        if (e.touches.length > 0) updateSplitHandle(e.touches[0]);
    }, { passive: true });

    window.addEventListener("touchend", () => {
        isDraggingSplit = false;
    });

    window.addEventListener("touchmove", (e) => {
        if (isDraggingSplit && e.touches.length > 0) {
            updateSplitHandle(e.touches[0]);
        }
    }, { passive: true });

    function updateSplitHandle(e) {
        const rect = splitBox.getBoundingClientRect();
        let posX = e.clientX - rect.left;
        posX = Math.max(0, Math.min(posX, rect.width));
        const pct = (posX / rect.width) * 100;

        splitClip.style.clipPath = `polygon(0 0, ${pct}% 0, ${pct}% 100%, 0 100%)`;
        splitHandle.style.left = `${pct}%`;
    }

    // Double click split handle to auto-center
    splitHandle.addEventListener("dblclick", () => {
        splitClip.style.clipPath = `polygon(0 0, 50% 0, 50% 100%, 0 100%)`;
        splitHandle.style.left = `50%`;
        showToast("Split comparison centered (50%)");
    });


    // 14. Sensor Specs Modal
    btnSensorModal.addEventListener("click", () => {
        modalSensor.classList.add("open");
    });

    btnModalClose.addEventListener("click", () => {
        modalSensor.classList.remove("open");
    });

    btnModalGotit.addEventListener("click", () => {
        modalSensor.classList.remove("open");
    });

    modalSensor.addEventListener("click", (e) => {
        if (e.target === modalSensor) {
            modalSensor.classList.remove("open");
        }
    });

    // 15. Modern Toast Notifications
    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast ${type === "error" ? "toast-error" : ""}`;
        toast.textContent = message;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(30px)";
            setTimeout(() => toast.remove(), 300);
        }, 2800);
    }
});
