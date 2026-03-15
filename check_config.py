import zipfile
import json
import os

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_model.keras")
with zipfile.ZipFile(model_path, "r") as z:
    with z.open("config.json") as f:
        config = json.load(f)

# Save formatted JSON so we can view it
with open("config_out.json", "w") as f:
    json.dump(config, f, indent=2)
