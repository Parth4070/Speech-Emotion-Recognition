# 🎤 Speech Emotion Recognition

An AI-powered web application that identifies the underlying emotion in spoken audio. Built with Python, Keras/TensorFlow, and Streamlit, this tool analyzes audio files (`.wav`) to predict emotions such as **Angry**, **Fearful**, **Happy**, **Neutral**, and **Sad**.

## 🚀 Features

- **Deep Learning Model:** Utilizes a custom Bidirectional LSTM architecture trained on MFCC (Mel-frequency cepstral coefficients) features extracted from speech.
- **Real-time Prediction:** Upload an audio file and get instant emotion predictions along with confidence scores.
- **Interactive UI:** A highly intuitive, responsive interface built with Streamlit.
- **Dynamic Standardization:** Integration with MongoDB to dynamically fetch pre-computed standardization statistics (mean & variance) for accurate feature scaling.

## 🛠️ Tech Stack

- **Audio Processing:** `librosa`, `soundfile`
- **Machine Learning:** `TensorFlow`, `Keras`
- **Backend/Database:** `PyMongo` (MongoDB)
- **Frontend/UI:** `Streamlit`
- **Utilities:** `NumPy`, `SciPy`

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- A MongoDB cluster (if you are fetching standardizations remotely)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
To connect to the database, you need to set up a Streamlit Secrets file locally.
Create a `.streamlit/secrets.toml` file in the root directory and add your MongoDB connection string:
```toml
MONGO_URI = "mongodb+srv://<username>:<password>@cluster.mongodb.net/..."
```

*(Note: During deployment on Streamlit Cloud, you manage this in the Advanced Settings -> Secrets section).*

### 4. Running the Application Locally
```bash
streamlit run app/streamlit_app.py
```
This will launch the app in your default web browser (typically at `http://localhost:8501`).

## 🧠 Model Architecture

The core of the application relies on an artificial neural network tailored for sequential data:
1. **Input:** MFCC features extracted from the `.wav` audio.
2. **Hidden Layers:** Two Bidirectional Long Short-Term Memory (LSTM) layers with interspersed Dropout layers to prevent overfitting.
3. **Output:** A Dense layer using Softmax activation to classify the input into one of the 5 distinct emotion categories.

## 📁 Repository Structure

```text
├── app/
│   └── streamlit_app.py     # Main Streamlit application
├── models/
│   └── best_model.keras     # Trained Keras model
├── .gitignore               # Ignored files (e.g., .env, secrets)
├── requirements.txt         # Python dependencies
└── README.md                # This file!
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.
