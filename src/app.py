import streamlit as st
import numpy as np
from PIL import Image
import os

# Set environment variables BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🤚",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #e94560, #ff6b6b, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .prediction-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
        text-align: center;
    }
    
    .prediction-label {
        font-size: 1rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 0.5rem;
    }
    
    .prediction-number {
        font-size: 6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    
    .confidence-text {
        font-size: 1.3rem;
        color: #4ade80;
        font-weight: 500;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_gesture_model():
    """Load the trained model"""
    try:
        model = load_model('/home/sengmeng/Desktop/I4 AMS/I4 AMS S1/AI/Course/Project_Sign_Language/src/notebook/alexnet_sign_language_model1.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image):
    """Preprocess image for prediction"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 224x224 (model's input size)
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Ensure the image has 3 channels (RGB)
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)  # Ensure 3 channels if not present
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_gesture(model, image):
    """Make prediction"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    return predicted_class, confidence, predictions[0]


# Main UI
st.markdown('<h1 class="main-title">🤚 Hand Gesture Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a hand gesture image to recognize the digit (0-9)</p>', unsafe_allow_html=True)

# Load model
model = load_gesture_model()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear image of a hand gesture"
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        if model is not None:
            with st.spinner("Analyzing..."):
                predicted_class, confidence, all_predictions = predict_gesture(model, image)
            
            st.markdown(f"""
            <div class="prediction-box">
                <p class="prediction-label">Predicted Digit</p>
                <p class="prediction-number">{predicted_class}</p>
                <p class="confidence-text">{confidence:.1f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Model not loaded!")

    # Show probabilities
    if model is not None:
        st.markdown("### 📊 All Predictions")
        chart_data = {f"Digit {i}": float(all_predictions[i] * 100) for i in range(10)}
        st.bar_chart(chart_data)

else:
    st.info("👆 Upload an image to get started")

# Footer
st.markdown("---")
cols = st.columns(10)
for i, col in enumerate(cols):
    col.markdown(f"<div style='text-align:center;background:rgba(255,255,255,0.05);padding:0.5rem;border-radius:8px;'><b style='color:#e94560;'>{i}</b></div>", unsafe_allow_html=True)