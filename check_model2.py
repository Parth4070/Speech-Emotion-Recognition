import zipfile
import json
import os

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_model.keras")
with zipfile.ZipFile(model_path, "r") as z:
    with z.open("metadata.json") as f:
        metadata = json.load(f)
    try:
        with z.open("config.json") as f:
            config = json.load(f)
            class_name = config.get("class_name", "UNKNOWN")
    except KeyError:
        class_name = "No config found"

with open("metadata_out.txt", "w") as f:
    f.write(f"Metadata: {metadata}\n")
    f.write(f"Class name: {class_name}\n")
