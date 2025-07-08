import io
import sqlite3
from datetime import datetime
from pathlib import Path

from loguru import logger
from PIL import Image

# Base directories (compatible with both local and Docker)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "corrections.db"

# Ensure folders exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"📁 Saving corrections to: {DB_PATH}")
logger.info(f"🖼️ Image directory: {IMAGES_DIR}")


def save_correction(file, label: int):
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = IMAGES_DIR / f"img_{timestamp}.png"

    # Save image to disk
    try:
        image = Image.open(io.BytesIO(file.file.read()))
        image.save(filename)
        logger.info(f"✅ Image saved: {filename}")
    except Exception as e:
        logger.error(f"❌ Failed to save image: {e}")
        raise

    # Save metadata to SQLite DB
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                label INTEGER,
                timestamp TEXT
            )
        """)

        # Insert correction
        cursor.execute(
            "INSERT INTO corrections (filename, label, timestamp) VALUES (?, ?, ?)",
            (str(filename), label, timestamp),
        )
        correction_id = cursor.lastrowid
        conn.commit()

        # Fetch inserted entry
        cursor.execute("SELECT * FROM corrections WHERE id = ?", (correction_id,))
        new_entry = cursor.fetchone()
        conn.close()

        logger.info(f"📥 Correction saved to DB: {new_entry}")

        return {
            "status": "correction saved",
            "entry": {
                "id": new_entry[0],
                "filename": new_entry[1],
                "label": new_entry[2],
                "timestamp": new_entry[3],
            },
        }

    except Exception as e:
        logger.error(f"❌ Failed to save correction to DB: {e}")
        raise
