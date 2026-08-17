"""SQLite-Anbindung für die Wissensdatenbank.

Ersetzt teil1/knowledge/*.txt als Datenquelle und protokolliert neue,
noch unbekannte Nutzerfragen.

Tabellen:
- documents: die früheren .txt-Dateien, jetzt als Zeilen in der DB
- questions: jede gestellte Frage, mit Verweis auf das gefundene
  Dokument (falls vorhanden) und die gegebene Antwort
"""

import re
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "knowledge.db"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"


def get_connection():
    """Öffnet eine SQLite-Verbindung. PRAGMA foreign_keys=ON, weil
    SQLite Fremdschlüssel sonst standardmäßig nicht prüft. """
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    """Legt die Tabellen an, falls sie noch nicht existieren.
    Sicher, mehrfach aufzurufen (CREATE TABLE IF NOT EXISTS) – wird bei
    jedem Programmstart ausgeführt, tut aber nach dem ersten Mal nichts.
    """
    connection = get_connection()
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            content TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            document_id INTEGER,
            answer_text TEXT,
            was_found_in_kb INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
        """
    )
    connection.commit()
    connection.close()

def migrate_txt_to_db():
    """Überträgt einmalig die .txt-Dateien in die documents-Tabelle.
    INSERT OR IGNORE + UNIQUE auf 'name' verhindert Duplikate bei
    erneutem Programmstart."""
    connection = get_connection()

    for file_path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            connection.execute(
                "INSERT OR IGNORE INTO documents (name, content) VALUES (?, ?)",
                (file_path.name, text),
            )

    connection.commit()
    connection.close()


def load_documents_from_db():
    """Ersetzt das alte load_documents() aus retrieval.py. Gleiche
    Struktur ({"name", "text"}, jetzt zusätzlich "id"), damit der Rest
    von retrieval.py unverändert weiterfunktioniert.
    """
    connection = get_connection()
    rows = connection.execute("SELECT id, name, content FROM documents").fetchall()  # Fetch all rows
    connection.close()

    return [
        {"id": row["id"], "name": row["name"], "text": row["content"]}
        for row in rows
    ]


def normalize_question(text):
    """Kleinschreibung + ohne Satzzeichen, damit z. B. 'Wie lange
    dauert die Lieferung?' und 'wie lange dauert die lieferung' als
    derselbe Vergleichswert erkannt werden.
    """
    normalized = text.strip().lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized


def question_exists(question_text):
    """Prüft per exaktem Abgleich des normalisierten Texts, ob die
    Frage schon in der Datenbank steht.
    """
    normalized = normalize_question(question_text)

    connection = get_connection()
    row = connection.execute(
        "SELECT id FROM questions WHERE normalized_text = ?",
        (normalized,),
    ).fetchone() # Fetch one
    connection.close()

    return row is not None


def log_question(question_text, document_id, answer_text, was_found):
    """Speichert eine neue Frage, falls sie noch nicht existiert (siehe
    question_exists()). Wird von chat.py nach jeder Antwort aufgerufen.
    """
    if question_exists(question_text):
        return

    connection = get_connection()
    connection.execute(
        """
        INSERT INTO questions
        (question_text, normalized_text, document_id, answer_text, was_found_in_kb)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            question_text,
            normalize_question(question_text),
            document_id,
            answer_text,
            int(was_found),
        ),
    )
    connection.commit()
    connection.close()
