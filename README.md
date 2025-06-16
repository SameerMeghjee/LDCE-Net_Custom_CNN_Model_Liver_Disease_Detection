# Liver Disease Detection With Custom CNN Model Using Ultrasound Images

## 🧠 Overview
LDCE-Net is a custom, lightweight deep learning model designed from scratch to classify liver fibrosis stages (Normal, Fibrosis, Cirrhosis) using grayscale ultrasound images. It is optimized to run on CPU-only systems and designed specifically for use in real-time clinical environments.

- ⚡ Fast and resource-efficient
- ✅ Trained from scratch (no pre-trained models)
- 🏥 Designed for clinical deployment (no GPU required)
- 🎛️ Simple GUI using Streamlit
- 📊 Training includes stratified sampling, dropout, and regularization to prevent overfitting

---

## 📂 Project Structure
```
LDCE-Net/
├── ldce_model.py             # Model definition (LDCE-Net with depthwise + attention)
├── training.py                  # Full training pipeline with plots and metrics
├── app.py                    # Streamlit GUI for image classification
├── ldce_model.pt             # Trained model weights
├── plots/                    # Accuracy, loss, and confusion matrix plots
├── README.md                 # Project documentation
```

---

## 📥 Dataset
This project uses the publicly available dataset from Kaggle:
🔗 [Liver Histopathology Fibrosis Ultrasound Images](https://www.kaggle.com/datasets/vibhingupta028/liver-histopathology-fibrosis-ultrasound-images)

After downloading, organize it into subfolders like:
```
Liver Ultrasounds/
├── F0-Normal/
├── F1-Fibrosis/
└── F2-Cirrhosis/
```

---

## 🧪 Training the Model
```bash
python training.py
```
This will:
- Train the model with stratified sampling
- Save best weights to `ldce_model.pt`
- Output plots to `plots/`

---

## 📊 Sample Training Metrics
- **Best validation accuracy**: ~87.6%
- **Train accuracy**: steadily rises to ~89%
- **No overfitting**: due to dropout + augmentations

---

## 🚀 GUI with Streamlit
```bash
streamlit run app.py
```
Upload any grayscale ultrasound image. The model will predict:
- F0-Normal
- F1-Fibrosis
- F2-Cirrhosis

---
