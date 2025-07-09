import os
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from app.model import build_model  # build_model(lr, dropout)

# Disable GPU for reproducibility
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train_and_validate_with_optuna(n_trials=10, epochs=20):
    """
    Search best hyperparameters with Optuna and retrain the final model.
    """
    print("📦 Loading MNIST dataset...")
    (x_train, y_train), _ = mnist.load_data()
    x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)

    print("🔀 Splitting dataset into train and validation sets...")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    print(f"🔍 Starting Optuna study with {n_trials} trials...")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)

        print(f"⚙️  Trial {trial.number}: lr={lr:.5f}, dropout={dropout:.2f}")

        model = build_model(lr=lr, dropout=dropout)
        es = EarlyStopping(patience=3, restore_best_weights=True)

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=[es],
            verbose=0,
        )

        best_val_acc = max(history.history["val_accuracy"])
        print(f"✅ Trial {trial.number} finished with best val accuracy: {best_val_acc:.4f}")
        return best_val_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"🏆 Best hyperparameters found: {study.best_params}")
    print(f"✨ Best validation accuracy during tuning: {study.best_value:.4f}")

    print("🔁 Retraining final model on full training data...")
    final_model = build_model(
        lr=study.best_params["lr"], dropout=study.best_params["dropout"]
    )
    final_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=0,
    )

    print("🧪 Evaluating final model...")
    _, val_acc = final_model.evaluate(x_val, y_val, verbose=0)
    print(f"✅ Final validation accuracy: {val_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    final_model.save("models/latest_model.h5")
    print("💾 Model saved to models/latest_model.h5")

    return val_acc


if __name__ == "__main__":
    print("🚀 Starting training script...")
    train_and_validate_with_optuna()
    print("✅ Script finished.")