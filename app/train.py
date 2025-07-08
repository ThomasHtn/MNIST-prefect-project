import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist

from app.model import build_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train_and_validate(model=None, epochs=10):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    if model is None:
        model = build_model()

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
    )

    val_loss, val_acc = model.evaluate(x_val, y_val)
    print(f"Validation Accuracy: {val_acc:.4f}")
    os.makedirs("models", exist_ok=True)
    print("📦 Dossier 'models/' prêt.")

    model.save("models/latest_model.h5")
    print("✅ Modèle sauvegardé.")
    return val_acc


if __name__ == "__main__":
    train_and_validate()
