"""
Pneumonia Detection - Streamlit Web App
Run with: streamlit run app/app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "models/best_model.pth"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Pneumonia X-Ray Detector",
    page_icon="🫁",
    layout="centered"
)

# ─────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    return None


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        output[0, pred_class].backward()
        pooled_grads = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (pooled_grads * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = output.softmax(dim=1)[0].detach().cpu().numpy()
        return cam, pred_class, probs


def overlay_heatmap(original_img, cam, alpha=0.5):
    heatmap = cm.jet(cam)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    original_np = np.array(original_img.resize((224, 224)))
    if original_np.ndim == 2:
        original_np = np.stack([original_np] * 3, axis=-1)
    return (alpha * heatmap + (1 - alpha) * original_np).astype(np.uint8)


# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(True)
    grad_cam = GradCAM(model, model.layer4[-1].conv3)
    cam, pred_class, probs = grad_cam.generate(input_tensor)
    overlay = overlay_heatmap(image, cam)
    return CLASS_NAMES[pred_class], probs, cam, overlay


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.markdown("""
Upload a chest X-ray image and the model will:
- **Classify** it as Normal or Pneumonia
- **Show** a Grad-CAM heatmap of what the model focused on
""")
st.divider()

model = load_model()

if model is None:
    st.warning("⚠️ Model file not found at `models/best_model.pth`. Please train the model first by running `src/train.py`.")
else:
    uploaded_file = st.file_uploader(
        "Upload a Chest X-Ray (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Analyzing X-ray..."):
            label, probs, cam, overlay = predict(image, model)

        # ─── Results ───
        is_pneumonia = label == "PNEUMONIA"
        color = "🔴" if is_pneumonia else "🟢"
        st.markdown(f"## {color} Prediction: **{label}**")

        col1, col2 = st.columns(2)
        col1.metric("Normal",    f"{probs[0]*100:.1f}%")
        col2.metric("Pneumonia", f"{probs[1]*100:.1f}%")

        if is_pneumonia:
            st.error("The model detected signs of Pneumonia. Please consult a doctor for proper diagnosis.")
        else:
            st.success("The model found no signs of Pneumonia.")

        st.divider()

        # ─── Visualization ───
        st.subheader("📊 Grad-CAM Visualization")
        st.caption("The heatmap shows which regions of the X-ray influenced the model's decision. Red = high attention.")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.patch.set_facecolor("#0e1117")

        axes[0].imshow(image.resize((224, 224)), cmap="gray")
        axes[0].set_title("Original", color="white")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", color="white")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", color="white")
        axes[2].axis("off")

        for ax in axes:
            ax.set_facecolor("#0e1117")

        plt.tight_layout()
        st.pyplot(fig)

        st.divider()
        st.caption("⚠️ **Disclaimer**: This tool is for educational purposes only and is NOT a substitute for professional medical diagnosis.")
