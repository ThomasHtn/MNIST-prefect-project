import os
import sqlite3
import sys

import pandas as pd
from prefect import flow, get_run_logger, task

# Import your train function (adjust the import if needed)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.train import train_and_validate

# Config
RETRAIN_THRESHOLD = 5  # number of corrections triggering retraining


@task
def load_corrections() -> pd.DataFrame:
    """
    Load new corrections from local SQLite database.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "..", "data", "corrections.db")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM corrections", conn)
    conn.close()
    return df


@task
def detect_failure(df: pd.DataFrame, threshold: int) -> list:
    """
    Identify classes with more than 'threshold' corrections.
    """
    class_counts = df["label"].value_counts()
    return class_counts[class_counts > threshold].index.tolist()


@flow(name="retraining-flow")
def retraining_flow():
    """
    Periodic retraining flow:
    - Load recent corrections
    - Detect problematic classes
    - Retrain model if necessary
    """
    logger = get_run_logger()

    df = load_corrections()
    problematic_classes = detect_failure(df, RETRAIN_THRESHOLD)

    if problematic_classes:
        logger.info(f"🔁 Retraining triggered for classes: {problematic_classes}")
        # train_and_validate should internally merge old dataset + new corrections
        val_acc = train_and_validate()
        logger.info(f"✅ New validation accuracy: {val_acc:.4f}")
    else:
        logger.info("✅ No retraining needed")


if __name__ == "__main__":
    # Schedule: every hour (3600 seconds)
    retraining_flow.serve(name="hourly-retraining", interval=3600)
