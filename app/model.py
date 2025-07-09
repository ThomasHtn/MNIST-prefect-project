from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam


def build_model(lr=0.0005, dropout=0.25):
    """
    Build and compile a CNN model for MNIST digit classification.
    """
    model = Sequential(
        [
            Conv2D(32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
            BatchNormalization(),
            MaxPooling2D(pool_size=2),
            Conv2D(64, kernel_size=3, activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=2),
            Conv2D(128, kernel_size=3, activation="relu"),
            BatchNormalization(),
            Flatten(),
            Dropout(dropout),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dense(10, activation="softmax"),  # 10 output classes
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
