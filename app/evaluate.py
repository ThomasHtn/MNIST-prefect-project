import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

# Load MNIST test data
(_, _), (x_test, y_test) = mnist.load_data()

# Normalize to [0, 1] and add channel dimension
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)

# Load trained model
model_path = "models/latest_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    print(f"📦 Model loaded from {model_path}")
else:
    raise FileNotFoundError(f"❌ Model not found at {model_path}")

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Accuracy: {test_acc:.4f}")
print(f"📉 Loss: {test_loss:.4f}")

# Predict classes
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Print classification report
print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(conf_matrix)

# Show some misclassified images
wrong_idx = np.where(y_pred != y_test)[0]

if len(wrong_idx) > 0:
    print(f"\n🔍 Number of misclassified samples: {len(wrong_idx)}")
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(wrong_idx[:16]):  # show first 16 mistakes
        plt.subplot(4, 4, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
        plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("👍 No misclassified samples found.")
