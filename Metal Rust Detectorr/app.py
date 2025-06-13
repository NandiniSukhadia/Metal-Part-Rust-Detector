import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Metal Rust Detector", layout="centered")
st.title("🧪 Metal Rust Detector")
st.write("Upload a photo of metal to detect rust.")

# Load model and labels
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def classify_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    result_idx = np.argmax(output_data)
    confidence = output_data[result_idx]
    return labels[result_idx], confidence

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, confidence = classify_image(image)
    st.markdown(f"### 🧾 Prediction: `{label}`")
    st.progress(int(confidence * 100))
    st.text(f"Confidence: {confidence*100:.2f}%")
