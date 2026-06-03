"""
Streamlit GUI for the Brain Tumor MRI Classifier.

The model stays behind the FastAPI service in main.py. This client sends image
uploads to that API and renders the prediction in a user-friendly interface.
"""

from __future__ import annotations

import os
from html import escape
from io import BytesIO
from typing import Any

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError


DEFAULT_API_URL = os.getenv("FASTAPI_URL", "https://brain-tumor-classifier-7jf5.onrender.com")
REQUEST_TIMEOUT_SECONDS = 90

CLASS_DETAILS = {
    "glioma": {
        "name": "Glioma",
        "description": "Tumor pattern associated with glial tissue.",
        "color": "#7c3aed",
    },
    "meningioma": {
        "name": "Meningioma",
        "description": "Tumor pattern associated with the meninges.",
        "color": "#0f766e",
    },
    "notumor": {
        "name": "No Tumor",
        "description": "No tumor pattern detected by the model.",
        "color": "#15803d",
    },
    "pituitary": {
        "name": "Pituitary",
        "description": "Tumor pattern associated with the pituitary region.",
        "color": "#b45309",
    },
}


def normalize_api_url(url: str) -> str:
    return url.strip().rstrip("/")


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health(api_url: str) -> dict[str, Any]:
    response = requests.get(f"{api_url}/health", timeout=8)
    response.raise_for_status()
    return response.json()


def predict_scan(api_url: str, file_name: str, file_type: str, image_bytes: bytes) -> dict[str, Any]:
    files = {
        "file": (
            file_name,
            image_bytes,
            file_type or "application/octet-stream",
        )
    }
    response = requests.post(
        f"{api_url}/predict",
        files=files,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"{response.status_code}: {detail}")

    return response.json()


def read_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes))
    return image.convert("RGB")


def get_uploaded_key(file_name: str, image_bytes: bytes) -> str:
    return f"{file_name}:{len(image_bytes)}:{hash(image_bytes)}"


def render_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: #f7f9fb;
                color: #172033;
            }

            .main .block-container {
                max-width: 1120px;
                padding-top: 2rem;
                padding-bottom: 2.5rem;
            }

            [data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid #e5e7eb;
            }

            .app-header {
                border-bottom: 1px solid #dbe4ef;
                margin-bottom: 1.4rem;
                padding-bottom: 1rem;
            }

            .app-kicker {
                color: #0f766e;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0;
                margin-bottom: 0.25rem;
                text-transform: uppercase;
            }

            .app-title {
                color: #111827;
                font-size: 2.25rem;
                font-weight: 760;
                line-height: 1.1;
                margin: 0;
            }

            .app-subtitle {
                color: #4b5563;
                font-size: 1rem;
                line-height: 1.55;
                margin: 0.65rem 0 0;
                max-width: 760px;
            }

            .status-pill {
                align-items: center;
                border-radius: 999px;
                display: inline-flex;
                font-size: 0.84rem;
                font-weight: 700;
                gap: 0.45rem;
                margin-bottom: 0.85rem;
                padding: 0.42rem 0.72rem;
            }

            .status-online {
                background: #dcfce7;
                color: #166534;
            }

            .status-offline {
                background: #fee2e2;
                color: #991b1b;
            }

            .panel {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 1rem;
            }

            .result-panel {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-left: 6px solid #0f766e;
                border-radius: 8px;
                padding: 1.15rem;
            }

            .result-eyebrow {
                color: #64748b;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0;
                margin-bottom: 0.4rem;
                text-transform: uppercase;
            }

            .result-label {
                color: #111827;
                font-size: 2rem;
                font-weight: 780;
                line-height: 1.1;
                margin-bottom: 0.35rem;
            }

            .result-copy {
                color: #4b5563;
                font-size: 0.95rem;
                line-height: 1.45;
                margin: 0;
            }

            .empty-panel {
                align-items: center;
                background: #ffffff;
                border: 1px dashed #cbd5e1;
                border-radius: 8px;
                color: #64748b;
                display: flex;
                min-height: 220px;
                justify-content: center;
                padding: 1.25rem;
                text-align: center;
            }

            .file-meta {
                color: #64748b;
                font-size: 0.86rem;
                margin-top: 0.45rem;
            }

            .safety-note {
                background: #fff7ed;
                border: 1px solid #fed7aa;
                border-radius: 8px;
                color: #7c2d12;
                font-size: 0.9rem;
                line-height: 1.45;
                margin-top: 1rem;
                padding: 0.85rem 1rem;
            }

            .stButton > button {
                border-radius: 8px;
                font-weight: 700;
                min-height: 2.8rem;
            }

            .stProgress > div > div > div > div {
                background-color: #0f766e;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-kicker">MRI Classification</div>
            <h1 class="app-title">Brain Tumor MRI Classifier</h1>
            <p class="app-subtitle">
                Upload a scan, send it to the FastAPI model service, and review
                the predicted class with confidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_api_status(api_url: str) -> tuple[bool, dict[str, Any] | None]:
    try:
        health = fetch_health(api_url)
    except requests.RequestException as exc:
        st.markdown(
            '<div class="status-pill status-offline">API offline</div>',
            unsafe_allow_html=True,
        )
        st.caption(str(exc))
        return False, None

    st.markdown(
        '<div class="status-pill status-online">API online</div>',
        unsafe_allow_html=True,
    )
    classes = ", ".join(health.get("classes", []))
    if classes:
        st.caption(f"Classes: {classes}")
    return True, health


def render_prediction(result: dict[str, Any]) -> None:
    label = str(result.get("label", "")).lower()
    confidence = float(result.get("confidence", 0.0))
    details = CLASS_DETAILS.get(
        label,
        {
            "name": label.title() or "Unknown",
            "description": "Prediction returned by the model service.",
            "color": "#0f766e",
        },
    )

    safe_name = escape(details["name"])
    safe_description = escape(details["description"])
    safe_color = escape(details["color"])

    st.markdown(
        f"""
        <div class="result-panel" style="border-left-color: {safe_color};">
            <div class="result-eyebrow">Prediction</div>
            <div class="result-label">{safe_name}</div>
            <p class="result-copy">{safe_description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.metric("Confidence", f"{confidence * 100:.1f}%")
    st.progress(max(0.0, min(confidence, 1.0)))


def main() -> None:
    st.set_page_config(
        page_title="Brain Tumor MRI Classifier",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_styles()
    render_header()

    with st.sidebar:
        st.header("FastAPI")
        api_url = normalize_api_url(st.text_input("URL", DEFAULT_API_URL))
        if st.button("Refresh status", use_container_width=True):
            fetch_health.clear()
        api_online, _ = render_api_status(api_url)

    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "uploaded_key" not in st.session_state:
        st.session_state.uploaded_key = None

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.subheader("Scan")
        uploaded_file = st.file_uploader(
            "MRI image",
            type=("jpg", "jpeg", "png"),
            accept_multiple_files=False,
        )

        image_bytes = None
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            uploaded_key = get_uploaded_key(uploaded_file.name, image_bytes)
            if uploaded_key != st.session_state.uploaded_key:
                st.session_state.uploaded_key = uploaded_key
                st.session_state.prediction = None

            try:
                image = read_image(image_bytes)
            except (UnidentifiedImageError, OSError):
                st.error("The selected file is not a valid image.")
                image_bytes = None
            else:
                st.image(image, use_container_width=True)
                st.markdown(
                    f"""
                    <div class="file-meta">
                        {escape(uploaded_file.name)} | {image.width} x {image.height}px
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="empty-panel">Awaiting scan</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.subheader("Result")
        analyze_clicked = st.button(
            "Analyze scan",
            type="primary",
            use_container_width=True,
            disabled=image_bytes is None,
        )

        if analyze_clicked and uploaded_file is not None and image_bytes is not None:
            with st.spinner("Analyzing scan"):
                try:
                    st.session_state.prediction = predict_scan(
                        api_url=api_url,
                        file_name=uploaded_file.name,
                        file_type=uploaded_file.type,
                        image_bytes=image_bytes,
                    )
                except (requests.RequestException, RuntimeError) as exc:
                    st.session_state.prediction = None
                    if not api_online:
                        st.error("FastAPI is not reachable from this URL.")
                    st.error(str(exc))

        if st.session_state.prediction:
            render_prediction(st.session_state.prediction)
        else:
            st.markdown(
                '<div class="empty-panel">Ready for analysis</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="safety-note">
                Decision support only. This model output is not a medical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
