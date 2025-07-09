import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from app.model import build_model

# Disable GPU (forces CPU usage)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train_and_validate(epochs=20):
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize images to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension (28x28 → 28x28x1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Split into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    # Create folder to save the model
    os.makedirs("models", exist_ok=True)

    model_path = "models/latest_model.h5"
    use_data_augmentation = not os.path.exists(model_path)

    if use_data_augmentation:
        print("✨ First training: using data augmentation")
        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
        )
        datagen.fit(x_train)
    else:
        print("🔁 Continuing training without data augmentation")

    # Build or load model
    if os.path.exists(model_path):
        print(f"📦 Loading existing model from {model_path}")
        model = load_model(model_path)
    else:
        print("🛠 Building new model")
        model = build_model()

    # Train model
    if use_data_augmentation:
        model.fit(
            datagen.flow(x_train, y_train, batch_size=64),
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        )
    else:
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        )

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(x_val, y_val)
    print(f"✅ Validation Accuracy: {val_acc:.4f}")

    # Save the trained model
    model.save(model_path)
    print(f"✅ Model saved at '{model_path}'")

    return val_acc


# Run training if this script is executed directly
if __name__ == "__main__":
    train_and_validate()
