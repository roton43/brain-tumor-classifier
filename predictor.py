"""
FastAPI runtime predictor.

The trained model is prepared once with prepare_model.py and saved to
artifacts/vit_brain_tumor.pkl. This module only loads that saved model at API
startup and reuses it for prediction requests.
"""

from __future__ import annotations

import io

import torch
from PIL import Image

from model_store import (
    CLASS_NAMES,
    MODEL_PICKLE_PATH,
    create_image_processor,
    load_model_pickle,
)


image_processor = create_image_processor()
model = load_model_pickle(MODEL_PICKLE_PATH)

print(f"[predictor] Pickled model loaded from {MODEL_PICKLE_PATH}. Classes: {CLASS_NAMES}")


def predict(image_bytes: bytes) -> dict:
    """
    Run inference on raw image bytes.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of any PIL-readable image format (JPEG, PNG, ...).

    Returns
    -------
    dict with keys:
        label      - predicted class name (str)
        confidence - softmax probability of the top class (float 0-1)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_idx = probs.argmax().item()

    return {
        "label": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
    }
