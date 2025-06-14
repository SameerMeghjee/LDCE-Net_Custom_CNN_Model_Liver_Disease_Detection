import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import LDCE_Net

import streamlit as st
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Liver Fibrosis Classifier", layout="centered")

# Settings
MODEL_PATH = "ldce_model.pt"
NUM_CLASSES = 3  
CLASS_NAMES = ["Normal (F0)", "Fibrosis (F1-F3)", "Cirrhosis (F4)"]

# Load Model
@st.cache_resource
def load_model():
    model = LDCE_Net(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

st.title("🧠 Liver Fibrosis Classification - LDCE-Net")
st.markdown("Upload an ultrasound image of the liver to classify fibrosis stage.")

# Display training accuracy curve if available
if os.path.exists("plots/accuracy_curve.png"):
    st.image("plots/accuracy_curve.png", caption="Model Training Accuracy Curve", use_container_width=True)

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Classify"):
        with st.spinner("Classifying..."):
            input_tensor = preprocess_image(image)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).detach().numpy()[0]
            pred_class = np.argmax(probs)
            st.success(f"Prediction: {CLASS_NAMES[pred_class]} ({probs[pred_class]*100:.2f}%)")

            # Display prediction probabilities as accuracy bar
            fig, ax = plt.subplots()
            ax.bar(CLASS_NAMES, probs * 100, color='green')
            ax.set_ylabel('Confidence (%)')
            ax.set_title('Prediction Confidence per Class')
            ax.set_ylim(0, 100)
            st.pyplot(fig)

            # Show detailed values
            st.write("### Class Probabilities:")
            for i, class_name in enumerate(CLASS_NAMES):
                st.write(f"{class_name}: {probs[i]*100:.2f}%")
