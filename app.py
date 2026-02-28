from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from gradcam import GradCAM, overlay_heatmap_on_image
from models.model import PneumoniaModel
from utils import CLASS_NAMES, load_checkpoint, load_image_for_inference


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_model(model_type: str, checkpoint_path: str, device_str: str) -> PneumoniaModel:
    device = torch.device(device_str)
    model = PneumoniaModel(model_type=model_type, pretrained=False).to(device)
    load_checkpoint(Path(checkpoint_path), model=model, map_location=device)
    model.eval()
    return model


def prepare_tensor_from_upload(uploaded_file) -> Tuple[Image.Image, torch.Tensor]:
    file_name = uploaded_file.name if uploaded_file.name else "uploaded_image.jpg"
    suffix = Path(file_name).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    return load_image_for_inference(temp_path)


def is_valid_xray(image: Image.Image) -> bool:
    """
    Lightweight heuristic input validation for chest X-ray-like images.
    - X-rays are typically grayscale (channels are similar).
    - Highly colorful images are likely out-of-domain.
    """
    rgb_image = image.convert("RGB")
    arr = np.asarray(rgb_image).astype(np.float32)

    if arr.ndim != 3 or arr.shape[2] != 3:
        return False

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Channel-difference signal: small for grayscale-like images.
    rg_diff = np.mean(np.abs(r - g))
    rb_diff = np.mean(np.abs(r - b))
    gb_diff = np.mean(np.abs(g - b))
    mean_channel_diff = float((rg_diff + rb_diff + gb_diff) / 3.0)

    # Color spread signal: high spread suggests non-medical colorful input.
    color_spread = np.std(arr, axis=2)
    mean_color_spread = float(np.mean(color_spread))

    grayscale_like = mean_channel_diff < 12.0
    low_color_diversity = mean_color_spread < 18.0
    return grayscale_like and low_color_diversity


def enable_mc_dropout(model: torch.nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def run_prediction_mc_dropout(
    model: PneumoniaModel,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[int, float, float, torch.Tensor]:
    enable_mc_dropout(model)

    mc_probs: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(20):
            logits = model(image_tensor.to(device))
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
            mc_probs.append(probs)

    stacked_probs = torch.stack(mc_probs, dim=0)
    mean_probs = stacked_probs.mean(dim=0)
    std_probs = stacked_probs.std(dim=0)

    confidence, pred_idx = torch.max(mean_probs, dim=0)
    uncertainty = float(std_probs[1].item())
    return int(pred_idx.item()), float(confidence.item()), uncertainty, mean_probs


def stratify_risk(pneumonia_probability: float) -> Tuple[str, str]:
    if pneumonia_probability > 0.75:
        return (
            "High Risk",
            "High pneumonia likelihood. Prioritize urgent clinical review and confirmatory assessment.",
        )
    if pneumonia_probability >= 0.50:
        return (
            "Moderate Risk",
            "Moderate pneumonia likelihood. Recommend prompt clinician review and close monitoring.",
        )
    return (
        "Low Risk",
        "Lower pneumonia likelihood. Continue standard clinical evaluation and monitor symptoms.",
    )


def render_uncertainty_status(uncertainty: float) -> None:
    if uncertainty <= 0.08:
        st.success(f"Confidence stability: Stable (low uncertainty) | {uncertainty * 100:.2f}%")
    else:
        st.error(f"Confidence stability: Unstable (high uncertainty) | {uncertainty * 100:.2f}%")


def render_probability_chart(probabilities: torch.Tensor) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(CLASS_NAMES, probabilities.numpy(), color=["#2b8cbe", "#de2d26"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probability Distribution")
    for idx, prob in enumerate(probabilities.tolist()):
        ax.text(idx, prob + 0.02, f"{prob:.2f}", ha="center", fontsize=10)
    st.pyplot(fig, clear_figure=True)


def main() -> None:
    st.set_page_config(page_title="AI for Sudan – Smart Medical Assistant", layout="wide")

    st.markdown(
        """
        <style>
            .main-title { font-size: 2rem; font-weight: 700; color: #114b5f; margin-bottom: 0.2rem; }
            .subtitle { font-size: 1.03rem; color: #4f6d7a; margin-bottom: 1.2rem; }
            .stAlert { border-radius: 0.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">AI for Sudan – Smart Medical Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Lightweight pneumonia screening support for low-resource healthcare settings using chest X-ray analysis.</div>',
        unsafe_allow_html=True,
    )

    device = get_device()

    with st.sidebar:
        st.header("Model Controls")
        model_type = st.selectbox("Model Type", options=["custom_cnn", "mobilenet_v2"], index=0)
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="outputs_smoke/checkpoints/best_model.pth",
            help="Path to a trained checkpoint (.pth)",
        )
        st.info(f"Running on: `{device.type.upper()}`")
        analyze_clicked = st.button("Analyze Image", type="primary", use_container_width=True)

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray Image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)
    else:
        st.warning("Upload an X-ray image to start analysis.")

    if analyze_clicked:
        if uploaded_file is None:
            st.error("Please upload an image before analyzing.")
            return
        if not checkpoint_path.strip():
            st.error("Please provide a valid checkpoint path.")
            return

        try:
            with st.spinner("Running model inference and generating Grad-CAM..."):
                original_image, image_tensor = prepare_tensor_from_upload(uploaded_file)

                if not is_valid_xray(original_image):
                    st.warning("This image does not appear to be a valid chest X-ray.")
                    st.info(
                        "This system is a decision-support tool and not a substitute for professional medical diagnosis."
                    )
                    return

                model = load_model(
                    model_type=model_type,
                    checkpoint_path=checkpoint_path.strip(),
                    device_str=str(device),
                )
                pred_idx, confidence, uncertainty, probs = run_prediction_mc_dropout(
                    model=model,
                    image_tensor=image_tensor,
                    device=device,
                )

                gradcam = GradCAM(model=model, target_layer=model.get_target_layer(), device=device)
                heatmap = gradcam.generate(input_tensor=image_tensor, class_idx=pred_idx)
                gradcam.remove_hooks()
                cam_overlay = overlay_heatmap_on_image(original_image, heatmap, alpha=0.45)

            predicted_class = CLASS_NAMES[pred_idx]
            pneumonia_probability = float(probs[1].item())
            risk_level, advisory_message = stratify_risk(pneumonia_probability)

            if predicted_class == "pneumonia":
                st.warning(f"Predicted class: **{predicted_class.upper()}**")
            else:
                st.success(f"Predicted class: **{predicted_class.upper()}**")

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Confidence Score", f"{confidence:.4f}")
            metric_col2.metric("Uncertainty", f"{uncertainty * 100:.2f}%")
            metric_col3.metric("Risk Level", risk_level)

            render_uncertainty_status(uncertainty)

            if uncertainty > 0.15:
                st.warning(
                    "⚠️ Model is uncertain. Input may not be a valid chest X-ray or requires human review."
                )

            if risk_level == "High Risk":
                st.error(f"Clinical Advisory: {advisory_message}")
            elif risk_level == "Moderate Risk":
                st.warning(f"Clinical Advisory: {advisory_message}")
            else:
                st.success(f"Clinical Advisory: {advisory_message}")

            st.info(
                "Higher uncertainty indicates lower model confidence and may require human review."
            )
            st.info(
                "This system is a decision-support tool and not a substitute for professional medical diagnosis."
            )

            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Original X-ray", use_container_width=True)
            with col2:
                st.image(cam_overlay, caption="Grad-CAM Overlay", use_container_width=True)

            render_probability_chart(probs)

            st.markdown(
                f"**Clinical Support Note:** The model focuses on image regions highlighted in the Grad-CAM map "
                f"to support a **{predicted_class}** prediction. This tool is for triage assistance and does not "
                "replace medical diagnosis."
            )

        except FileNotFoundError as err:
            st.error(f"File error: {err}")
        except RuntimeError as err:
            st.error(f"Model runtime error: {err}")
        except Exception as err:
            st.error(f"Unexpected error during analysis: {err}")


if __name__ == "__main__":
    main()
