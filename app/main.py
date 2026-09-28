
from contextlib import asynccontextmanager
from io import BytesIO

import torch

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.model_loader import load_model


# Shared model resources
resources = {}


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    model, processor, class_names = load_model()

    resources["model"] = model
    resources["processor"] = processor
    resources["class_names"] = class_names

    yield

    resources.clear()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ViT LoRA INT8 API",
    description=(
        "EuroSAT image classification using a "
        "LoRA-fine-tuned and INT8-quantized ViT."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "ViT LoRA INT8 API",
        "status": "running",
        "documentation": "/docs"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():
    ready = all(
        key in resources
        for key in ("model", "processor", "class_names")
    )

    if not ready:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready"
        )

    return {
        "status": "healthy",
        "model": "ViT + LoRA + INT8",
        "device": "cpu",
        "classes": len(resources["class_names"])
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if "model" not in resources:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/webp"
    ):
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG, PNG or WebP image"
        )

    try:
        image_bytes = file.file.read(10 * 1024 * 1024 + 1)

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Maximum file size is 10 MB"
            )

        image = Image.open(BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image"
        )

    model = resources["model"]
    processor = resources["processor"]
    class_names = resources["class_names"]

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.inference_mode():
        outputs = model(
            pixel_values=inputs["pixel_values"]
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=-1
        )

    class_id = int(prediction.item())

    return {
        "filename": file.filename,
        "class_id": class_id,
        "label": class_names[class_id],
        "confidence": round(
            float(confidence.item()), 4
        ),
        "device": "cpu",
        "quantization": "dynamic_int8"
    }