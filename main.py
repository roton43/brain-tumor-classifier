"""
main.py
FastAPI application for Brain Tumor MRI Classification.

Endpoints:
    GET  /health   - Server status and model info
    POST /predict  - Upload an MRI image, receive label + confidence

Run (development):
    fastapi dev main.py

Run (production):
    fastapi run main.py

Test via Swagger UI:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import predictor                          # loads model at import time

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description=(
        "Fine-tuned Vision Transformer (google/vit-base-patch16-224) "
        "that classifies brain MRI scans into four categories: "
        "glioma, meningioma, notumor, pituitary."
    ),
    version="1.0.0",
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model: str
    classes: list[str]


class PredictionOutput(BaseModel):
    label: str
    confidence: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, summary="Server health check")
def health():
    """Returns server status, model name, and supported class labels."""
    return HealthResponse(
        status="ok",
        model="ViT fine-tuned on Brain Tumor MRI (google/vit-base-patch16-224)",
        classes=predictor.CLASS_NAMES,
    )


@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Classify a brain MRI image",
)
async def predict(file: UploadFile = File(..., description="Brain MRI image (JPEG or PNG)")):
    """
    Upload a brain MRI image and receive the predicted tumor class with confidence score.

    - **label**: one of `glioma`, `meningioma`, `notumor`, `pituitary`
    - **confidence**: softmax probability of the predicted class (0.0 – 1.0)
    """
    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG or PNG image.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predictor.predict(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return PredictionOutput(**result)
