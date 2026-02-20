import os
import numpy as np
import librosa
from tqdm import tqdm

EMOTION_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful"
}

def extract_emotion(filename):
    """
    Example filename:
    03-01-05-01-02-01-12.wav
                ↑
                05 = emotion
    """
    emotion_code = filename.split("-")[2]

    if emotion_code in EMOTION_MAP:
        return EMOTION_MAP[emotion_code]
    else:
        return None

def extract_mfcc(file_path, max_len=130):
    """
    Extract MFCC features and pad/truncate to fixed length.
    """

    audio, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # Transpose so shape = (time_steps, features)
    mfcc = mfcc.T

    # Padding or truncating
    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:max_len]

    return mfcc

def build_dataset(data_path):
    X = []
    y = []
    actors = []

    for actor_folder in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor_folder)

        if not os.path.isdir(actor_path):
            continue

        actor_id = actor_folder 

        for file in os.listdir(actor_path):
            if file.endswith(".wav"):

                emotion = extract_emotion(file)
                if emotion is None:
                    continue

                file_path = os.path.join(actor_path, file)
                mfcc = extract_mfcc(file_path)

                X.append(mfcc)
                y.append(emotion)
                actors.append(actor_id)

    return np.array(X), np.array(y), np.array(actors)

if __name__ == "__main__":

    data_path = "../data/raw"

    X, y, actors = build_dataset(data_path)

    print("Feature shape:", X.shape)
    print("Labels shape:", y.shape)

    processed_path = "../data/processed"
    os.makedirs(processed_path, exist_ok=True)

    np.save(os.path.join(processed_path, "X.npy"), X)
    np.save(os.path.join(processed_path, "y.npy"), y)
    np.save(os.path.join(processed_path, "actors.npy"), actors)

    print("Saved X and y successfully.")