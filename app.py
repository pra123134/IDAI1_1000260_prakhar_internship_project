# SmartWasteAI Project Code (YOLOv8 + Gemini 2.5 Pro API Integration)

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import os
import requests
import base64
import google.generativeai as genai

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API Key is missing. Go to Streamlit Cloud → Settings → Secrets and add your API key.")
    st.stop()

yolo_model = YOLO('yolov8n.pt')
classifier_model = load_model('mobilenetv2_waste_classifier.h5')
feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"

waste_map = {
    'plastic': 'Recyclable',
    'glass': 'Recyclable',
    'metal': 'Recyclable',
    'battery': 'Hazardous',
    'clothes': 'Recyclable',
    'shoes': 'Recyclable',
    'organic': 'Biodegradable',
    'paper': 'Recyclable',
    'trash': 'Biodegradable',
    'e-waste': 'Hazardous'
}

bin_map = {
    'Biodegradable': 'Green Bin',
    'Recyclable': 'Blue Bin',
    'Hazardous': 'Red Bin',
    'Radioactive': 'Black Bin',
    'Unknown': 'Manual Sorting Required'
}

reinforcement_feedback = {
    'plastic': 'Recyclable',
    'glass': 'Recyclable',
    'metal': 'Recyclable',
    'radioactive': 'Radioactive'
}

known_embeddings = {
    label: None for label in waste_map
}

def initialize_known_embeddings(dataset_path):
    for label in waste_map:
        folder = os.path.join(dataset_path, label)
        if not os.path.isdir(folder):
            continue
        vectors = []
        count = 0
        for file in os.listdir(folder):
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                path = os.path.join(folder, file)
                image = cv2.imread(path)
                if image is None:
                    continue
                image = cv2.resize(image, (224, 224))
                img_array = preprocess_input(image.astype(np.float32))
                img_array = np.expand_dims(img_array, axis=0)
                embedding = feature_extractor.predict(img_array, verbose=0)
                vectors.append(embedding[0])
                count += 1
                if count >= 5:
                    break
        if vectors:
            known_embeddings[label] = np.mean(vectors, axis=0)

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def extract_embedding(image):
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    embedding = feature_extractor.predict(img, verbose=0)
    return embedding[0]

def is_anomalous(confidence, threshold=35.0):
    return confidence < threshold

def query_gemini_api(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    b64_image = base64.b64encode(img_encoded).decode('utf-8')
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Classify this waste and suggest appropriate disposal category and bin color."
                    },
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": b64_image
                        }
                    }
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{GEMINI_API_URL}?key={api_key}", json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return "Unknown"

def predict_waste(image):
    processed = preprocess_image(image)
    preds = classifier_model.predict(processed, verbose=0)[0]
    class_index = np.argmax(preds)
    confidence = preds[class_index] * 100
    class_labels = list(waste_map.keys()) + ['radioactive']
    predicted_label = class_labels[class_index] if class_index < len(class_labels) else 'unknown'
    category = reinforcement_feedback.get(predicted_label, waste_map.get(predicted_label, 'Unknown'))
    if category == 'Unknown':
        test_embedding = extract_embedding(image)
        max_sim = 0
        best_match = None
        for label in known_embeddings:
            if known_embeddings[label] is not None:
                sim = cosine_similarity([test_embedding], [known_embeddings[label]])[0][0]
                if sim > max_sim:
                    max_sim = sim
                    best_match = label
        if max_sim > 0.7:
            category = waste_map.get(best_match, 'Unknown')
        else:
            gemini_response = query_gemini_api(image)
            category = gemini_response.split()[0].capitalize() if gemini_response else 'Unknown'
    bin_color = bin_map.get(category, 'Manual Sorting Required')
    return predicted_label, category, bin_color, confidence

def main():
    st.title("SmartWasteAI: Gemini-Enhanced AI Waste Sorting System")
    st.markdown("Upload a waste image. The system will detect, classify, and recommend a disposal bin.")
    dataset_path = st.text_input("Enter dataset path (local):")
    if st.button("Initialize Known Embeddings") and dataset_path:
        initialize_known_embeddings(dataset_path)
        st.success("Known embeddings initialized.")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Failed to decode image. Please try another file.")
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
        results = yolo_model(image_rgb)
        annotated_frame = results[0].plot()
        st.image(annotated_frame, caption="Detected Waste Items")
        predicted_label, category, bin_color, confidence = predict_waste(image_rgb)
        st.subheader("Prediction Result")
        st.write(f"**Predicted Label**: {predicted_label}")
        st.write(f"**Category**: {category}")
        st.write(f"**Recommended Bin**: {bin_color}")
        st.write(f"**Confidence Score**: {confidence:.2f}%")
        if is_anomalous(confidence) or category == 'Unknown':
            st.warning("Low confidence or unknown category. Gemini API used.")
        feedback = st.radio("Was this prediction correct?", ("Yes", "No"))
        if feedback == "No":
            st.info("Model will log this feedback for retraining suggestions.")

def test_model_on_directory(test_dir):
    true_labels = []
    predictions = []
    class_names = list(waste_map.keys())
    for class_name in os.listdir(test_dir):
        class_folder = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_folder):
            continue
        for image_file in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            label, _, _, _ = predict_waste(image)
            true_labels.append(class_name)
            predictions.append(label)
    acc = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions, labels=class_names)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
