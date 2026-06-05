from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import UnidentifiedImageError
from pydantic import BaseModel
import predictor                         

app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description=(
        "Fine-tuned Vision Transformer (google/vit-base-patch16-224) "
        "that classifies brain MRI scans into four categories: "
        "glioma, meningioma, notumor, pituitary."
    ),
    version="1.0.0",
)


class HealthResponse(BaseModel):
    status: str
    model: str
    classes: list[str]


class RootResponse(BaseModel):
    status: str
    message: str
    endpoints: dict[str, str]


class PredictionOutput(BaseModel):
    label: str
    confidence: float


@app.get("/", response_model=RootResponse, summary="API root")
def root():
    """Returns a small landing response so the base API URL is not a 404."""
    return RootResponse(
        status="ok",
        message="Brain Tumor MRI Classifier API is running. Open /docs for Swagger UI.",
        endpoints={
            "docs": "/docs",
            "health": "/health",
            "predict": "POST /predict",
        },
    )


@app.get("/health", response_model=HealthResponse, summary="Server health check")
def health():
    """Returns server status, model name, and supported class labels."""
    return HealthResponse(
        status="ok",
        model="ViT fine-tuned on Brain Tumor MRI (google/vit-base-patch16-224)",
        classes=predictor.get_class_names(),
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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception as exc:
        import traceback
        print("[ERROR] Prediction failed:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return PredictionOutput(**result)