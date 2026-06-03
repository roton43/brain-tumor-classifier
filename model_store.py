"""
Model artifact helpers for the Brain Tumor MRI Classifier.

Run prepare_model.py once to convert the trained .pt checkpoint into a pickled
model artifact. The FastAPI app then loads only the .pkl file at startup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import urlretrieve

import torch
from PIL import Image
from transformers import ViTConfig, ViTForImageClassification

try:
    from transformers import ViTImageProcessorPil as ViTImageProcessor
except ImportError:  # Transformers 4.x
    from transformers import ViTImageProcessor


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
WEIGHTS_PATH = ARTIFACTS_DIR / "vit_brain_tumor.pt"
MODEL_PICKLE_PATH = ARTIFACTS_DIR / "vit_brain_tumor.pkl"
CLASSES_PATH = ARTIFACTS_DIR / "class_names.json"
MODEL_WEIGHTS_URL = os.getenv("MODEL_WEIGHTS_URL", "").strip()


with open(CLASSES_PATH, encoding="utf-8") as f:
    CLASS_NAMES: list[str] = json.load(f)

ID2LABEL = {idx: label for idx, label in enumerate(CLASS_NAMES)}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}


def create_image_processor() -> ViTImageProcessor:
    return ViTImageProcessor(
        do_resize=True,
        size={"height": 224, "width": 224},
        resample=Image.Resampling.BILINEAR,
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )


def create_model_config() -> ViTConfig:
    return ViTConfig(
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


def ensure_weights_file(path: Path = WEIGHTS_PATH) -> None:
    if path.exists():
        return

    if not MODEL_WEIGHTS_URL:
        raise FileNotFoundError(
            f"Model weights not found at {path}. Add the checkpoint locally or "
            "set MODEL_WEIGHTS_URL to a direct download URL before preparing "
            "the pickled model."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[model_store] Downloading model weights to {path}")
    try:
        urlretrieve(MODEL_WEIGHTS_URL, path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError("Failed to download model weights from MODEL_WEIGHTS_URL") from exc


def load_checkpoint(path: Path = WEIGHTS_PATH) -> dict:
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


def remap_legacy_vit_keys(state_dict: dict, target_state_dict: dict) -> dict:
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


def build_model_from_weights() -> ViTForImageClassification:
    ensure_weights_file()
    model = ViTForImageClassification(create_model_config())
    state_dict = load_checkpoint()
    state_dict = remap_legacy_vit_keys(state_dict, model.state_dict())
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_model_pickle(path: Path = MODEL_PICKLE_PATH) -> Path:
    model = build_model_from_weights()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)
    print(f"[model_store] Saved pickled model to {path}")
    return path


def load_model_pickle(path: Path = MODEL_PICKLE_PATH) -> ViTForImageClassification:
    if not path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {path}. Run `python prepare_model.py` "
            "before starting the FastAPI app."
        )

    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(path, map_location="cpu")

    model.eval()
    return model
