"""
predictor.py
Loads the fine-tuned ViT model and image processor once at startup,
then exposes a predict(image_bytes) function for FastAPI.

Artifacts required:
    artifacts/vit_brain_tumor.pt   - model state_dict (trained on Kaggle T4 GPU)
    artifacts/class_names.json     - class label list in ImageFolder order
"""

import io
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import ViTConfig, ViTForImageClassification

try:
    from transformers import ViTImageProcessorPil as ViTImageProcessor
except ImportError:  # Transformers 4.x
    from transformers import ViTImageProcessor

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "artifacts" / "vit_brain_tumor.pt"
CLASSES_PATH = BASE_DIR / "artifacts" / "class_names.json"

# ── Load class names ───────────────────────────────────────────────────────────
with open(CLASSES_PATH, encoding="utf-8") as f:
    CLASS_NAMES: list[str] = json.load(f)

ID2LABEL = {idx: label for idx, label in enumerate(CLASS_NAMES)}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}

# ── Load image processor (handles resize to 224×224 + normalisation) ──────────
image_processor = ViTImageProcessor(
    do_resize=True,
    size={"height": 224, "width": 224},
    resample=Image.Resampling.BILINEAR,
    do_rescale=True,
    rescale_factor=1.0 / 255.0,
    do_normalize=True,
    image_mean=[0.5, 0.5, 0.5],
    image_std=[0.5, 0.5, 0.5],
)

config = ViTConfig(
    image_size=224,
    patch_size=16,
    num_channels=3,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    qkv_bias=True,
    encoder_stride=16,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

model = ViTForImageClassification(config)


def _load_checkpoint(path: Path) -> dict:
    """Load a state_dict saved by either plain PyTorch or Trainer-style code."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

    return checkpoint


def _remap_legacy_vit_keys(state_dict: dict, target_state_dict: dict) -> dict:
    """
    Transformers 5 renamed ViT encoder keys. Accept checkpoints saved with the
    Transformers 4 names and translate them only when the target model needs it.
    """
    checkpoint_uses_legacy_keys = any(
        key.startswith("vit.encoder.layer.") for key in state_dict
    )
    target_uses_new_keys = any(key.startswith("vit.layers.") for key in target_state_dict)

    if not checkpoint_uses_legacy_keys or not target_uses_new_keys:
        return state_dict

    replacements = (
        ("vit.encoder.layer.", "vit.layers."),
        (".attention.attention.query.", ".attention.q_proj."),
        (".attention.attention.key.", ".attention.k_proj."),
        (".attention.attention.value.", ".attention.v_proj."),
        (".attention.output.dense.", ".attention.o_proj."),
        (".intermediate.dense.", ".mlp.fc1."),
        (".output.dense.", ".mlp.fc2."),
    )

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        remapped[new_key] = value

    return remapped


# map_location='cpu' is mandatory because weights may have been saved on a GPU.
state_dict = _load_checkpoint(WEIGHTS_PATH)
state_dict = _remap_legacy_vit_keys(state_dict, model.state_dict())
model.load_state_dict(state_dict)

# Switch to eval mode: disables dropout, fixes BatchNorm statistics.
model.eval()

print(f"[predictor] Model loaded. Classes: {CLASS_NAMES}")


# ── Inference function ─────────────────────────────────────────────────────────
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
    # Decode bytes -> PIL Image, force RGB (handles grayscale DICOM exports etc.)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ViTImageProcessor resizes to 224x224 and normalises with ViT defaults.
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert raw logits to probabilities
    probs    = torch.softmax(outputs.logits, dim=-1)[0]
    pred_idx = probs.argmax().item()

    return {
        "label":      CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
    }
