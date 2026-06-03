# Brain Tumor MRI Classifier

## Overview
Brain Tumor MRI Classifier is a local inference application for classifying brain
MRI images into four categories:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

The project has two runtime parts:

- `main.py`: FastAPI service that loads the fine-tuned ViT model and exposes
  prediction endpoints.
- `app.py`: Streamlit GUI that connects to the FastAPI URL, lets users upload
  an MRI image, previews the scan, and displays the predicted class with
  confidence.

The model runs from local artifacts, so inference does not require downloading
weights from Hugging Face at startup.

## Features
- Fine-tuned Vision Transformer classifier.
- FastAPI backend for model health checks and image prediction.
- Streamlit frontend with image upload, preview, API status, confidence display,
  and configurable FastAPI URL.
- Local CPU inference support.
- Git ignore rules for Python caches, virtual environments, Streamlit secrets,
  and large model checkpoints.

## Model Details
| Property | Value |
|---|---|
| Architecture | `google/vit-base-patch16-224` fine-tuned for 4 classes |
| Dataset | Brain Tumor MRI Dataset from Kaggle |
| Classes | `glioma`, `meningioma`, `notumor`, `pituitary` |
| Test Accuracy | ~97% after 6 epochs |
| Optimizer | AdamW, learning rate `2e-5` |
| Scheduler | Linear warmup plus decay |

Dataset source:
`kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset`

## Project Structure
```text
brain-tumor-classifier/
|-- app.py                    # Streamlit GUI client
|-- main.py                   # FastAPI application
|-- predictor.py              # Model loading, preprocessing, and inference
|-- requirements.txt          # Python dependencies
|-- submission.txt
|-- artifacts/
|   |-- class_names.json      # Class label mapping
|   `-- vit_brain_tumor.pt    # Fine-tuned model weights, ignored by git
|-- training/
|   `-- vit_finetuning.ipynb  # Training notebook
|-- .gitignore
`-- README.md
```

## Installation
Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Required local artifacts:

```text
artifacts/class_names.json
artifacts/vit_brain_tumor.pt
```

`vit_brain_tumor.pt` is intentionally ignored by git because it is a large
checkpoint file. Keep it in the `artifacts/` directory before running the API.

## Run the FastAPI Model Service
Start the backend first:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Useful API URLs:

- Swagger docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Run the Streamlit GUI
In a second terminal, start the Streamlit app:

```bash
streamlit run app.py
```

Open:

```text
http://127.0.0.1:8501
```

The GUI uses this FastAPI URL by default:

```text
http://127.0.0.1:8000
```

To use a different backend, either edit the URL in the Streamlit sidebar or set
`FASTAPI_URL` before launching Streamlit:

```bash
FASTAPI_URL=http://127.0.0.1:8000 streamlit run app.py
```

## API Endpoints
| Method | Endpoint | Input | Output |
|---|---|---|---|
| GET | `/health` | None | Server status, model name, class list |
| POST | `/predict` | JPEG or PNG image file as `file` | Predicted label and confidence |

Example prediction response:

```json
{
  "label": "glioma",
  "confidence": 0.9821
}
```

## Quick API Test
After starting FastAPI, test the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Test image prediction:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@path/to/mri-image.png"
```

## Troubleshooting
If the Streamlit app shows the API as offline, make sure the FastAPI service is
running on the same URL shown in the Streamlit sidebar.

If model loading fails, confirm that `artifacts/vit_brain_tumor.pt` exists and
that `artifacts/class_names.json` contains the four class names.

If the upload is rejected, use a JPEG or PNG file.

## Safety Note
This project is for educational and decision-support use only. The output is not
a medical diagnosis and should not replace review by a qualified clinician.

## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- FastAPI
- Uvicorn
- Streamlit
- Pillow
- Requests
