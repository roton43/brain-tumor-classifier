"""
FastAPI runtime predictor.

The trained model is prepared once with prepare_model.py and saved to
artifacts/vit_brain_tumor.pkl. This module only loads that saved model at API
startup and reuses it for prediction requests.
"""

# predictor.py
import io
import torch
from PIL import Image
from model_store import CLASS_NAMES, MODEL_PICKLE_PATH, create_image_processor, load_model_pickle

# Move model loading inside a function
_model = None

def get_model():
    global _model
    if _model is None:
        print("[predictor] Loading model into memory...")
        _model = load_model_pickle(MODEL_PICKLE_PATH)
    return _model

image_processor = create_image_processor()

def predict(image_bytes: bytes) -> dict:
    model = get_model() # Load only when needed
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
