import os
import sys
import numpy as np
import librosa
import streamlit as st

# Suppress verbose TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

LABEL_MAPPING = ["angry", "fearful", "happy", "neutral", "sad"]

# Streamlit Page Config
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤", layout="centered")

@st.cache_resource
def load_ser_model():
    """Load and cache the Keras model."""
    model_path = os.path.join(BASE_DIR, "models", "best_model.keras")
    if not os.path.exists(model_path):
        st.error(f"Error: Model not found at {model_path}. Ensure you have run training.")
        return None
    return load_model(model_path, compile=False)

@st.cache_data
def load_training_stats():
    """Fetch training data statistics from MongoDB for standardization."""
    from pymongo import MongoClient

    # When deployed on Streamlit Cloud, you set MONGO_URI in the Secrets management UI
    # Locally, it will read it from a .streamlit/secrets.toml file if created
    if "MONGO_URI" not in st.secrets:
        st.error("Error: Database connection secret (MONGO_URI) not found in Streamlit Secrets.")
        return None, None

    try:
        client = MongoClient(st.secrets["MONGO_URI"], serverSelectionTimeoutMS=5000)
        db = client["SpeechEmotionAI"]
        collection = db["ModelStats"]

        # Fetch the pre-computed document we uploaded
        doc = collection.find_one({"_id": "standardization_params"})
        
        if not doc:
            st.error("Error: Standardization parameters not found in MongoDB database.")
            return None, None
            
        # Convert nested lists back into numpy arrays exactly as the model expects them
        mean = np.array(doc["mean"])
        std = np.array(doc["std"])
        
        return mean, std
        
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return None, None

def extract_mfcc(audio_bytes, max_len=130):
    """
    Extract MFCC features from raw audio bytes and pad/truncate to fixed length.
    """
    import io
    import soundfile as sf
    try:
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = mfcc.T

    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:max_len]

    return mfcc

def predict_emotion(audio_bytes, model, mean, std):
    mfcc_feature = extract_mfcc(audio_bytes)
    if mfcc_feature is None:
        return None, None

    # Needs to be shape (1, timesteps, features)
    X_input = np.expand_dims(mfcc_feature, axis=0)

    # Standardize input
    X_input = (X_input - mean) / std

    # Predict
    predictions = model.predict(X_input)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_idx]

    if predicted_class_idx < len(LABEL_MAPPING):
        predicted_emotion = LABEL_MAPPING[predicted_class_idx]
    else:
        predicted_emotion = f"Unknown (ID: {predicted_class_idx})"
        
    return predicted_emotion, confidence

def main():
    st.title("🎤 Speech Emotion Recognition")
    st.write("Upload a `.wav` audio file and let the AI predict the emotion in the speech!")
    
    # Load model and stats
    model = load_ser_model()
    mean, std = load_training_stats()
    
    if model is None or mean is None:
        st.stop()

    uploaded_file = st.file_uploader("Choose a `.wav` audio file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Predict Emotion", type="primary", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                audio_bytes = uploaded_file.read()
                emotion, confidence = predict_emotion(audio_bytes, model, mean, std)
                
                if emotion is not None:
                    st.success(f"**Predicted Emotion:** {emotion.upper()}")
                    st.info(f"**Confidence:** {confidence * 100:.2f}%")
                    
                    # Create a nice confidence bar chart across all labels optionally
                    # Here we just show the top prediction
                    st.progress(float(confidence), text=f"Confidence: {confidence * 100:.2f}%")

if __name__ == "__main__":
    main()
