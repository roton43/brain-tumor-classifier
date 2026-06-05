import io
import torch
from PIL import Image
from model_store import get_class_names, MODEL_PICKLE_PATH, create_image_processor, load_model_pickle

_model = None
_image_processor = None


def get_model():
    global _model
    if _model is None:
        print("[predictor] Loading model into memory...")
        _model = load_model_pickle(MODEL_PICKLE_PATH)
    return _model


def get_image_processor():
    global _image_processor
    if _image_processor is None:
        _image_processor = create_image_processor()
    return _image_processor


def predict(image_bytes: bytes) -> dict:
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Empty image bytes received")

    model = get_model()
    image_processor = get_image_processor()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image: {exc}")

    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_idx = probs.argmax().item()
    class_names = get_class_names()

    return {
        "label": class_names[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
    }