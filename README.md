# Pneumonia Detection from Chest X-Rays

A deep learning project that detects **Pneumonia** from chest X-ray images using a fine-tuned **ResNet50** with **Grad-CAM** explainability and a **Streamlit** web app for live demos.

> **Resume bullet:**  
> *Built a pneumonia detection system using fine-tuned ResNet50 on 5,800 chest X-rays achieving ~92% accuracy, with Grad-CAM heatmaps highlighting infected lung regions and a Streamlit diagnostic web app.*

---

## Project Structure

```
pneumonia-detection/
├── src/
│   ├── train.py          # Model training + evaluation
│   └── gradcam.py        # Grad-CAM visualization
├── app/
│   └── app.py            # Streamlit web app
├── models/               # Saved model weights (generated after training)
├── data/                 # Dataset (download from Kaggle)
├── requirements.txt
└── README.md
```

---

## Dataset

Download from Kaggle:  
🔗 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Extract into:
```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train.py
```
This will:
- Fine-tune ResNet50 on your chest X-ray dataset
- Save the best model to `models/best_model.pth`
- Plot training history and confusion matrix

### 3. Generate Grad-CAM visualizations
```bash
python src/gradcam.py
```
Update the `test_images` list in the script with your actual image paths.

### 4. Launch the Streamlit app
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser and upload any chest X-ray image.

---

## Model Architecture

| Component         | Details                                |
|-------------------|----------------------------------------|
| Base Model        | ResNet50 (pretrained on ImageNet)      |
| Fine-tuned Layers | `layer4` + custom FC head             |
| FC Head           | Dropout → Linear(2048→256) → ReLU → Dropout → Linear(256→2) |
| Loss Function     | CrossEntropyLoss                       |
| Optimizer         | Adam (lr=0.001)                        |
| Scheduler         | StepLR (step=4, gamma=0.5)            |
| Class Imbalance   | WeightedRandomSampler                  |

---

## Results

<img width="396" height="143" alt="image" src="https://github.com/user-attachments/assets/430aacc2-361b-4434-89fc-7eedfab363f2" />

> **Note**: Recall is prioritized over Precision in medical diagnosis — missing a sick patient (False Negative) is more dangerous than a false alarm (False Positive).

---

## 🔍 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of the X-ray that the model focused on when making its prediction.

- 🔴 **Red regions** = high attention (suspicious areas)
- 🔵 **Blue regions** = low attention

This is critical in medical AI — models must be **interpretable** for clinical adoption (FDA guidelines for AI medical devices require explainability).

### Snapshot of Grad-CAM

<img width="1283" height="500" alt="Figure_3" src="https://github.com/user-attachments/assets/ed9a184f-5ef8-4047-870b-b98afe9c21fe" />

<img width="1283" height="500" alt="Figure_4" src="https://github.com/user-attachments/assets/72b21f3f-1d74-4a5b-aff7-8c37e4efdb93" />

---

##  Streamlit App Features

- Upload any chest X-ray image (JPG/PNG)
- Get instant Normal/Pneumonia prediction with confidence scores
- View Grad-CAM heatmap and overlay visualization
- Clean, dark-themed UI

### Snapshot of Streamlit

<img width="361" height="490" alt="Screenshot 2026-05-16 131042" src="https://github.com/user-attachments/assets/8834309f-bd6d-48df-9d2d-54e3aaa2d37d" />

<img width="319" height="485" alt="Screenshot 2026-05-16 131156" src="https://github.com/user-attachments/assets/96f761e4-9427-489b-a1ac-fa453dd6d307" />

---

##  Key Design Decisions

**Why ResNet50?**
- Strong pretrained features from ImageNet
- Residual connections help with gradient flow on medical images
- Widely used baseline in medical imaging literature

**Why WeightedRandomSampler?**
- Dataset has ~3x more PNEUMONIA than NORMAL images(Normal=1341(25%), Pneumonia=3875(75%))
- Without balancing, model would be biased toward PNEUMONIA
- WeightedRandomSampler ensures balanced batches during training

**Why Grad-CAM?**
- Black-box models are unacceptable in healthcare
- Grad-CAM provides visual explanation without modifying the model
- Clinicians can verify if the model is looking at the right regions

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `torchvision` · `Streamlit` · `scikit-learn` · `Matplotlib` · `OpenCV` · `Pillow`

---
