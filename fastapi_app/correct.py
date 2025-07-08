import io
import os
import sqlite3
from datetime import datetime
from PIL import Image

DB_PATH = "../data/corrections.db"

def save_correction(file, label: int):
    os.makedirs("../data/images", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"../data/images/img_{timestamp}.png"

    # Save image to disk
    image = Image.open(io.BytesIO(file.file.read()))
    image.save(filename)

    # Save metadata to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            label INTEGER,
            timestamp TEXT
        )
    """)
    cursor.execute(
        "INSERT INTO corrections (filename, label, timestamp) VALUES (?, ?, ?)",
        (filename, label, timestamp),
    )
    correction_id = cursor.lastrowid  # Récupère l'ID de la dernière insertion
    conn.commit()

    # Récupère l'entrée tout juste ajoutée
    cursor.execute("SELECT * FROM corrections WHERE id = ?", (correction_id,))
    new_entry = cursor.fetchone()
    conn.close()

    return {
        "status": "correction saved",
        "entry": {
            "id": new_entry[0],
            "filename": new_entry[1],
            "label": new_entry[2],
            "timestamp": new_entry[3],
        },
    }