import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
import base64
from io import BytesIO

# Custom CSS for layout and styles
st.markdown("""
    <style>
        header[data-testid="stHeader"] {
            background-color: #f0f0f0;
            color: black;
        }

        .stApp {
            background-color: white;
            color: black;
        }

        h1, h2, h3, h4, h5, h6, p, span {
            color: black !important;
        }

        .stFileUploader {
            background-color: white !important;
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 10px;
        }

        button[kind="primary"] {
            background-color: #4a90e2 !important;
            color: white !important;
            border: 2px solid #90ee90 !important; /* light green border */
            border-radius: 8px !important;
            padding: 0.5em 1em !important;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }

        .centered {
            display: flex;
            justify-content: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }

        .centered img {
            border-radius: 12px;
            width: 300px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)


# Define class labels
CLASS_NAMES = [
    "Acral_Lentiginous_Melanoma",
    "Healthy_Nail",
    "Onychogryphosis",
    "blue_finger",
    "clubbing",
    "pitting"
]

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("my_model.keras")
    return model

# Load the model
with st.spinner("Loading model..."):
    model = load_model()

# Streamlit UI
st.title("🩺 Nail Disease Detection App")
st.write("Upload a nail image, and the model will classify the disease.")

# File uploader
file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Preprocessing function
def import_and_predict(image, model):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image / 255.0
    img_reshape = np.expand_dims(image, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

# Process uploaded image
if file:
    image = Image.open(file)

    # Convert image to base64 for HTML rendering
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    # Display image in the center
    st.markdown(f"""
        <div class="centered">
            <img src="data:image/png;base64,{img_b64}" alt="Uploaded Image"/>
        </div>
    """, unsafe_allow_html=True)

    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])

    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Display prediction
    st.success(f"🔍 **Prediction:** {predicted_class}")
    # st.info(f"📊 **Confidence:** {confidence:.2f}%")
