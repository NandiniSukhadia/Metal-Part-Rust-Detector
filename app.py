import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Metal Part Rust Detector", layout="centered")

st.title("🧪 Metal Part Rust Detector")
st.write("Upload an image of a metal part to check if it’s **Rusty** or **Clean**.")

@st.cache_resource
def load_model():
    if not os.path.exists("model.tflite"):
        st.error("🚨 model.tflite file not found in the root directory.")
        st.stop()
    try:
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Failed to load model.tflite: {e}")
        st.stop()

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image.astype(np.float32), axis=0)

def predict(image):
    input_data = preprocess(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

input_mode = st.radio("Choose Input Method", ["Upload Image", "Use Camera"])
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
else:
    uploaded_file = st.camera_input("Take a photo")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image", use_column_width=True)
    try:
        pred = predict(image)
        classes = ["Rusty", "Clean"]
        confidence = np.max(pred)
        label = classes[np.argmax(pred)]

        st.markdown(f"### 🔍 Prediction: `{label}`")
        st.progress(int(confidence * 100))
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
