import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DATABASE_PATH = Path(__file__).with_name("review_analyzer.db")

def initialize_database():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review TEXT NOT NULL,
                label TEXT NOT NULL,
                score INTEGER NOT NULL CHECK(score BETWEEN -1 AND 1),
                theme TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

def save_analysis(review, result, source):
    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO analyses (review, label, score, theme, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                review,
                result["label"],
                result["score"],
                result["theme"],
                source,
                created_at,
            )
        )

        return cursor.lastrowid

def get_analyses():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, review, label, score, theme, source, created_at
            FROM analyses
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]