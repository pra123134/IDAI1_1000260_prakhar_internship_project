import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Set Streamlit page configuration
st.set_page_config(page_title="Simple YOLOv8 Waste Detector", layout="centered")

# Load YOLOv8 model only once
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Object detection using YOLO
def detect_objects(image):
    results = model(np.array(image))
    return results[0].plot(show_conf=False)

# Streamlit App
st.title("🗑️ Simple YOLOv8 Waste Detector")
st.markdown("Upload an image and YOLOv8 will detect objects.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.subheader("🔍 Detection Result")
        result_img = detect_objects(image)

        fig, ax = plt.subplots()
        ax.imshow(result_img)
        ax.axis("off")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error while processing image: {e}")
