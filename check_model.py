import zipfile
import json
import os

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_model.keras")
with zipfile.ZipFile(model_path, "r") as z:
    with z.open("metadata.json") as f:
        metadata = json.load(f)
        print("Metadata:", metadata)
    try:
        with z.open("config.json") as f:
            config = json.load(f)
            print("Config keys:", config.keys())
    except KeyError:
        print("No config.json found")
