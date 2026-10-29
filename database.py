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
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                owner_id TEXT NOT NULL
            )
            """      
        )
    
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(documents)")
        }

        if "owner_id" not in columns:
            connection.execute("ALTER TABLE documents ADD COLUMN owner_id TEXT")
            connection.execute(
                "UPDATE documents SET owner_id = ? WHERE owner_id IS NULL",
                ("alice",),
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
    
def save_document(file_name, file_path, owner_id):
    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents (file_name, file_path, created_at, owner_id)
            VALUES (?, ?, ?, ?)
            """,
            (file_name, file_path, created_at, owner_id)
        )

        return cursor.lastrowid
    
def get_documents():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, file_name, file_path, created_at, owner_id
            FROM documents
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]