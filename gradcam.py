"""
Grad-CAM Visualization for Pneumonia Detection
Generates heatmaps showing which regions of the X-ray the model focused on
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms, models
from PIL import Image
import cv2
import os

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_model.pth"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model(model_path):
    import torch.nn as nn
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ─────────────────────────────────────────────
# GRAD-CAM IMPLEMENTATION
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None

        # Hook into the target layer
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backprop for target class
        output[0, class_idx].backward()

        # Pool gradients
        pooled_grads = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weight activations
        cam = (pooled_grads * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx, output.softmax(dim=1)[0].detach().cpu().numpy()


# ─────────────────────────────────────────────
# OVERLAY HEATMAP ON IMAGE
# ─────────────────────────────────────────────
def overlay_heatmap(original_img, cam, alpha=0.5):
    heatmap = cm.jet(cam)[:, :, :3]  # RGB
    heatmap = (heatmap * 255).astype(np.uint8)
    original_np = np.array(original_img.resize((224, 224)))
    if original_np.ndim == 2:
        original_np = np.stack([original_np] * 3, axis=-1)
    overlay = (alpha * heatmap + (1 - alpha) * original_np).astype(np.uint8)
    return overlay


# ─────────────────────────────────────────────
# PREDICT + VISUALIZE SINGLE IMAGE
# ─────────────────────────────────────────────
def predict_and_visualize(image_path, model, save_path=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    original_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_img).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(True)

    # Attach Grad-CAM to the last conv layer of ResNet50
    grad_cam = GradCAM(model, model.layer4[-1].conv3)

    cam, pred_class, probs = grad_cam.generate(input_tensor)
    overlay = overlay_heatmap(original_img, cam)

    # ─── PLOT ───
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Prediction: {CLASS_NAMES[pred_class]}  |  "
        f"Normal: {probs[0]*100:.1f}%  |  Pneumonia: {probs[1]*100:.1f}%",
        fontsize=13, fontweight="bold",
        color="red" if pred_class == 1 else "green"
    )

    axes[0].imshow(original_img.resize((224, 224)), cmap="gray")
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Model Focus)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
    return CLASS_NAMES[pred_class], probs


# ─────────────────────────────────────────────
# BATCH VISUALIZATION — show N examples
# ─────────────────────────────────────────────
def visualize_batch(image_paths, model, labels=None, save_dir="models/gradcam"):
    os.makedirs(save_dir, exist_ok=True)
    for i, path in enumerate(image_paths):
        label_str = f"_true={labels[i]}" if labels else ""
        save_path = os.path.join(save_dir, f"gradcam_{i}{label_str}.png")
        pred, probs = predict_and_visualize(path, model, save_path=save_path)
        print(f"[{i+1}/{len(image_paths)}] {os.path.basename(path)} → {pred}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = load_model(MODEL_PATH)

    # Example: visualize a few test images
    # Replace these paths with actual images from your test set
    test_images = [
    r"C:\Users\SUSHANT\OneDrive\Desktop\Project\pneumonia-detection\chest_xray\test\NORMAL\IM-0001-0001.jpeg",

    r"C:\Users\SUSHANT\OneDrive\Desktop\Project\pneumonia-detection\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg",
    ]

    existing = [p for p in test_images if os.path.exists(p)]
    if not existing:
        print("⚠️  No test images found. Update the paths in test_images list above.")
    else:
        visualize_batch(existing, model)
