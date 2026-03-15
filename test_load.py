import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_model.keras")
try:
    load_model(model_path, compile=False)
    print("Model loaded successfully")
except Exception as e:
    import traceback
    traceback.print_exc()
