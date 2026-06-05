import json
from pathlib import Path
from model_store import save_model_pickle, ARTIFACTS_DIR


def prepare_artifacts():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    classes_path = ARTIFACTS_DIR / "class_names.json"
    
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f)
    print(f"[prepare_model] Saved class names to {classes_path}")

    save_model_pickle()
    print("[prepare_model] All artifacts ready.")


if __name__ == "__main__":
    prepare_artifacts()