
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

model = load_model("../models/latest_model.h5")

def predict_digit(file):
    image = Image.open(io.BytesIO(file.file.read())).convert("L").resize((28, 28))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    prediction = model.predict(image)
    pred_class = int(np.argmax(prediction))
    return {"prediction": pred_class}
