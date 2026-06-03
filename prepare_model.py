"""
Prepare the saved model artifact used by the FastAPI app.

This script builds the ViT model from artifacts/vit_brain_tumor.pt once and
saves it as artifacts/vit_brain_tumor.pkl. The API loads the .pkl file directly.
"""

from __future__ import annotations

import argparse

from model_store import MODEL_PICKLE_PATH, save_model_pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the pickled model artifact.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the .pkl file even if it already exists.",
    )
    args = parser.parse_args()

    if MODEL_PICKLE_PATH.exists() and not args.force:
        print(f"[prepare_model] Saved model already exists: {MODEL_PICKLE_PATH}")
        return

    save_model_pickle(MODEL_PICKLE_PATH)


if __name__ == "__main__":
    main()
