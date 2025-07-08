
import sqlite3

def init_db():
    conn = sqlite3.connect("data/feedback.db")
    cursor = conn.cursor()
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS corrections (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               timestamp TEXT,
               label INTEGER,
               image_path TEXT
           )'''
    )
    conn.commit()
    conn.close()
