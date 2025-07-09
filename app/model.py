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
        [  # Create a Keras Sequential model (a linear stack of layers)
            Conv2D(32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
            # First convolutional layer: 32 filters of size 3x3, ReLU activation
            # input_shape specifies 28x28 grayscale images (1 channel)
            BatchNormalization(),
            # Normalize activations to speed up training and reduce overfitting
            MaxPooling2D(pool_size=2),
            # Downsample feature maps using 2x2 pooling to reduce spatial dimensions
            Conv2D(64, kernel_size=3, activation="relu"),
            # Second convolutional layer: 64 filters of size 3x3, ReLU activation
            BatchNormalization(),
            # Normalize again for training stability
            MaxPooling2D(pool_size=2),
            # Downsample again with another 2x2 pooling
            Conv2D(128, kernel_size=3, activation="relu"),
            # Third convolutional layer: 128 filters of size 3x3
            BatchNormalization(),
            # Normalize activations
            Flatten(),
            # Flatten the feature maps into a 1D vector to feed into dense layers
            Dropout(dropout),
            # Apply dropout to prevent overfitting; rate controlled by the dropout parameter
            Dense(128, activation="relu"),
            # Fully connected layer with 128 units and ReLU activation
            BatchNormalization(),
            # Normalize again before final output
            Dense(10, activation="softmax"),
            # Output layer: 10 units (one for each digit 0–9), softmax for probability distribution
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
