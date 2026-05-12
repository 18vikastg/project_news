"""SQLite helpers. Path defaults next to this package; override with NEWS_DB_PATH."""

import os
import sqlite3
from pathlib import Path

DATABASE_PATH = os.environ.get(
    "NEWS_DB_PATH", str(Path(__file__).resolve().parent / "news_analysis.db")
)


def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            kannada_text TEXT NOT NULL,
            english_translation TEXT,
            is_fake BOOLEAN,
            confidence REAL,
            category TEXT,
            category_confidence REAL,
            summary TEXT,
            analysis_method TEXT,
            translation_quality TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """
    )

    conn.commit()
    conn.close()


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn
