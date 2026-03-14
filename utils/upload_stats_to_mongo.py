import os
import sys
import numpy as np
from pymongo import MongoClient
import urllib.parse
from dotenv import load_dotenv

# Suppress verbose TensorFlow logging if needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def upload_stats():
    # Load environment variables if they exist in a .env file locally
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    # The URI should look like: mongodb+srv://<username>:<password>@cluster0...
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("Error: Please set MONGO_URI in your environment or a .env file.")
        print("Example .env file content: MONGO_URI=mongodb+srv://admin:pass@cluster...")
        sys.exit(1)

    print("Connecting to MongoDB...")
    client = MongoClient(mongo_uri)
    
    # We'll use a database called 'SpeechEmotionAI' and a collection called 'ModelStats'
    db = client["SpeechEmotionAI"]
    collection = db["ModelStats"]

    train_data_path = os.path.join(BASE_DIR, "data", "train", "X_train.npy")
    if not os.path.exists(train_data_path):
        print(f"Error: local training data not found at {train_data_path}.")
        sys.exit(1)

    print("Loading local X_train.npy...")
    X_train = np.load(train_data_path)
    
    print("Calculating mean and std...")
    # Calculate exactly like in predict.py / train_model.py
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True)

    # MongoDB requires basic Python types, so convert Numpy arrays to nested lists
    mean_list = float(np.mean(X_train)) # The normalization calculation historically used per-feature or global.
    # Looking at the original logic `np.mean(X_train, axis=(0, 1), keepdims=True)`
    # This results in an array of shape (1, 1, 40) if 40 features
    mean_array_list = mean.tolist()
    std_array_list = std.tolist()

    document = {
        "_id": "standardization_params", # Fixed ID so we can easily overwrite and fetch it
        "description": "Mean and Std for MFCC standardization",
        "mean": mean_array_list,
        "std": std_array_list,
    }

    print("Uploading to MongoDB...")
    # Use replace_one with upsert=True to insert if it doesn't exist, or replace if it does
    collection.replace_one({"_id": "standardization_params"}, document, upsert=True)
    
    print("Successfully uploaded model statistics to MongoDB!")

if __name__ == "__main__":
    upload_stats()
