import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import sqlite3

import pandas as pd
from prefect import flow, get_run_logger, task

from app.train import train_and_validate


@task
def load_corrections():
    # Load corrections from local SQLite database
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "..", "data", "corrections.db")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM corrections", conn)
    conn.close()

    return df


@task
def detect_failure(df: pd.DataFrame):
    # Detect overrepresented classes (heuristic: >5 corrections)
    class_counts = df["label"].value_counts()
    return class_counts[class_counts > 5].index.tolist()


@flow(name="retraining-flow")
def retraining_flow():
    logger = get_run_logger()

    # Load corrections and identify failure-prone classes
    df = load_corrections()
    problematic_classes = detect_failure(df)

    if problematic_classes:
        logger.info(f"🔁 Retraining triggered for classes: {problematic_classes}")
        val_acc = train_and_validate()
        logger.info(f"✅ New validation accuracy: {val_acc:.4f}")
    else:
        logger.info("✅ No retraining needed")


# Local mode
if __name__ == "__main__":
    retraining_flow.serve(name="every-2-minutes", interval=120)

# if __name__ == "__main__":
#     retraining_flow()
