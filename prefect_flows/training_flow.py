import os
import sqlite3

import pandas as pd
from prefect import flow, get_run_logger, task

from app.train import train_and_validate


@task
def load_corrections():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "..", "data", "corrections.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM corrections", conn)
    conn.close()
    return df


@task
def detect_failure(df):
    failed_classes = df["label"].value_counts()
    return failed_classes[failed_classes > 5].index.tolist()


@flow
def retraining_flow():
    logger = get_run_logger()

    df = load_corrections()
    problematic_classes = detect_failure(df)

    if problematic_classes:
        logger.info(f"Réentraînement pour les classes : {problematic_classes}")
        val_acc = train_and_validate()
        logger.info(f"✅ Validation accuracy : {val_acc:.4f}")
    else:
        logger.info("✅ Nothing to do")


if __name__ == "__main__":
    retraining_flow.serve(name="every-2-minutes", interval=120)
