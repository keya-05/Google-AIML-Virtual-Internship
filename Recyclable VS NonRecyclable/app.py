import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Load trained model
model = tf.keras.models.load_model("recycle_classifier.h5")

# Load class label mapping
with open("class_indices.json") as f:
    class_indices = json.load(f)

# Reverse the class index mapping
idx_to_class = {v: k for k, v in class_indices.items()}

# Define recyclable classes (all lowercase and simple)
RECYCLABLE_CLASSES = {"glass", "metal", "paper", "plastic"}

st.title("♻ Recyclable vs Non-Recyclable Classifier")
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).resize((224, 224)).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_id = int(np.argmax(prediction))
    predicted_class_name = idx_to_class[predicted_class_id]

    # Clean predicted class name for comparison
    cleaned_class = predicted_class_name.strip().lower()

    # ✅ Use partial match in case class name includes extra tokens (e.g., "paper (R)")
    is_recyclable = any(recyclable in cleaned_class for recyclable in RECYCLABLE_CLASSES)

    # Show result
    label = "♻ Recyclable" if is_recyclable else "🚫 Non-Recyclable"
    st.success(f"This item is: **{label}**")
    st.info(f"Predicted class: `{predicted_class_name}`")
