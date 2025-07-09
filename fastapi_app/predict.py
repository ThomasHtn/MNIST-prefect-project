import io
import os
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from tensorflow.keras.models import load_model

# Force CPU usage (ignore GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define model path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "latest_model.h5"

# Load model at startup
try:
    model = load_model(MODEL_PATH)
    logger.info(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None


def predict_digit(file):
    """
    Predict the digit from an uploaded file.
    """
    if model is None:
        logger.error("❌ Model not loaded")
        return {"error": "Model not loaded"}

    try:
        # Read image and preprocess
        image = Image.open(io.BytesIO(file.file.read())).convert("L")
        image_array = preprocess_image(image)
        image_array = np.expand_dims(image_array, axis=0)  # shape: (1, 28, 28, 1)

        # Make prediction
        predictions = model.predict(image_array)
        predicted_digit = int(np.argmax(predictions[0]))

        logger.info(f"🔢 Predicted digit: {predicted_digit}")
        return {"prediction": predicted_digit}

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return {"error": "Prediction failed"}


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prepare image for digit recognition:
    - invert colors
    - crop to bounding box
    - resize to 28x28
    - normalize to [0,1]
    """
    img_array = np.array(image)

    # Invert: white digit on black background
    img_array = 255 - img_array

    # Find bounding box (ignore low-level noise)
    coords = np.column_stack(np.where(img_array > 30))
    if coords.size:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        img_array = img_array[y_min : y_max + 1, x_min : x_max + 1]

    # Resize to 28x28
    img_resized = Image.fromarray(img_array).resize((28, 28), Image.LANCZOS)

    # Normalize and add channel dimension
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # shape: (28, 28, 1)

    return img_array
