from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam


def build_model(lr=0.001, dropout=0.3):
    model = Sequential(
        [
            Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(2),
            Conv2D(64, 3, activation="relu"),
            MaxPooling2D(2),
            Flatten(),  # Flatten to 1D
            Dropout(dropout),  # Dropout for regularization
            Dense(128, activation="relu"),  # Fully connected layer
            Dense(10, activation="softmax"),  # Output layer (10 classes)
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
