import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Ignore GPU pour dev

import io
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from tensorflow.keras.models import load_model

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "latest_model.h5"

# Load model at startup
try:
    model = load_model(MODEL_PATH)
    logger.info(f"📦 Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load model from {MODEL_PATH}: {e}")
    model = None


def predict_digit(file):
    if model is None:
        logger.error("❌ No model loaded, cannot predict")
        return {"error": "Model not loaded"}

    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("L")
        image = image.resize((28, 28))
        image_array = np.array(image).astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=(0, -1))  # (1, 28, 28, 1)

        predictions = model.predict(image_array)
        predicted_digit = int(np.argmax(predictions[0]))

        logger.info(f"🔢 Predicted digit: {predicted_digit}")
        return {"prediction": predicted_digit}
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return {"error": "Prediction failed"}
