import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

st.set_page_config(page_title="Simple YOLOv8 Waste Detector", layout="centered")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

def detect_objects(image):
    results = model(image)
    return results[0].plot(show_conf=False)

def main():
    st.title("🗒️ Simple Waste Detection with YOLOv8")
    st.write("Upload an image to detect waste objects using YOLOv8 Nano model.")

    uploaded_file = st.file_uploader("📄 Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            st.image(image, caption="📸 Uploaded Image", use_column_width=True)

            st.subheader("🔍 YOLOv8 Detection Output")
            result_img = detect_objects(image_np)

            fig, ax = plt.subplots()
            ax.imshow(result_img)
            ax.axis("off")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Failed to process image: {e}")

if __name__ == "__main__":
    main()
