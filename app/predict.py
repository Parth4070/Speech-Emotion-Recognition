import os
import sys
import numpy as np
import librosa
import argparse

# Suppress verbose TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

LABEL_MAPPING = ["angry", "fearful", "happy", "neutral", "sad"]


def extract_mfcc(file_path, max_len=130):
    """
    Extract MFCC features and pad/truncate to fixed length (exactly like in dataset_build.py).
    """
    try:
        audio, sr = librosa.load(file_path, sr=22050)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        sys.exit(1)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = mfcc.T

    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:max_len]

    return mfcc


def main(file_path):
    print(f"Loading and processing {file_path}...")
    
    # 1. Extract feature
    mfcc_feature = extract_mfcc(file_path)
    
    # Needs to be shape (1, timesteps, features)
    X_input = np.expand_dims(mfcc_feature, axis=0)

    # 2. Load training data to calculate exact mean and std for standardisation
    train_data_path = os.path.join(BASE_DIR, "data", "train", "X_train.npy")
    if not os.path.exists(train_data_path):
        print(f"Error: Training data not found at {train_data_path}. Ensure you have run data generation.")
        sys.exit(1)

    X_train = np.load(train_data_path)
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True)

    # Standardize input
    X_input = (X_input - mean) / std

    # 3. Load model
    model_path = os.path.join(BASE_DIR, "models", "best_model.keras")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Ensure you have run training.")
        sys.exit(1)

    print("Loading model. This might take a few seconds...")
    model = load_model(model_path)

    # 4. Predict
    predictions = model.predict(X_input)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_idx]

    if predicted_class_idx < len(LABEL_MAPPING):
        predicted_emotion = LABEL_MAPPING[predicted_class_idx]
    else:
        predicted_emotion = f"Unknown (ID: {predicted_class_idx})"

    print("\n" + "="*40)
    print(f"Prediction Results:")
    print(f"  Predicted Emotion : {predicted_emotion.upper()}")
    print(f"  Confidence        : {confidence * 100:.2f}%")
    print("="*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from a an audio file.")
    parser.add_argument("file_path", type=str, help="Path to the .wav audio file.")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: The provided file path does not exist: {args.file_path}")
        sys.exit(1)

    main(args.file_path)
