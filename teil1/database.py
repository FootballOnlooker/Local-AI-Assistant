"""SQLite-Anbindung für die Wissensdatenbank.

Ersetzt die bisherige Datenquelle (teil1/knowledge/*.txt) durch eine
richtige Datenbank, und protokolliert neue, bisher unbekannte Fragen der
Nutzer – beide Anforderungen aus der Zusatzaufgabe.

Tabellen:
- documents: die ehemaligen .txt-Dateien, jetzt als Zeilen in der DB
- questions: jede vom Nutzer gestellte Frage, mit Verweis auf das
  gefundene Dokument (falls eines gefunden wurde) und der gegebenen Antwort
"""

import re
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "knowledge.db"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"


def get_connection():
    """Öffnet eine Verbindung zur SQLite-Datei
    """
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    """Legt die Tabellen an, falls sie noch nicht existieren.
    Sicher, mehrfach aufzurufen (CREATE TABLE IF NOT EXISTS) – wird bei
    jedem Programmstart ausgeführt, tut aber nach dem ersten Mal nichts.
    """
    connection = get_connection()
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents
        (
            id
            INTEGER
            PRIMARY
            KEY
            AUTOINCREMENT,
            name
            TEXT
            NOT
            NULL
            UNIQUE,
            content
            TEXT
            NOT
            NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS questions
        (
            id
            INTEGER
            PRIMARY
            KEY
            AUTOINCREMENT,
            question_text
            TEXT
            NOT
            NULL,
            normalized_text
            TEXT
            NOT
            NULL,
            document_id
            INTEGER,
            answer_text
            TEXT,
            was_found_in_kb
            INTEGER
            NOT
            NULL
            DEFAULT
            0,
            created_at
            TEXT
            NOT
            NULL
            DEFAULT (
            datetime
        (
            'now'
        )),
            FOREIGN KEY
        (
            document_id
        ) REFERENCES documents
        (
            id
        )
            )
        """
    )
    connection.commit()
    connection.close()


def migrate_txt_to_db():
    """Einmalige Migration: liest alle .txt-Dateien aus knowledge/ und
    fügt sie in die documents-Tabelle ein. INSERT OR IGNORE zusammen mit
    UNIQUE auf 'name' sorgt dafür, dass beim erneuten Programmstart keine
    Duplikate entstehen – bereits vorhandene Namen werden übersprungen.
    """
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
    """Ersetzt das alte load_documents() aus retrieval.py, das die
    Textdateien direkt gelesen hat. Gibt dieselbe Struktur zurück
    ({"name": ..., "text": ...}, jetzt zusätzlich mit "id"), damit
    chunk_documents() und der Rest von retrieval.py unverändert
    weiterfunktionieren.
    """
    connection = get_connection()
    rows = connection.execute("SELECT id, name, content FROM documents").fetchall()  # Fetch all rows
    connection.close()

    return [
        {"id": row["id"], "name": row["name"], "text": row["content"]}
        for row in rows
    ]


def normalize_question(text):
    """Entfernt Satzzeichen und vereinheitlicht Groß-/Kleinschreibung,
    damit z. B. 'Wie lange dauert die Lieferung?' und
    'wie lange dauert die lieferung' als dieselbe Frage erkannt werden.
    Gleiches Prinzip wie get_template_group() in teil2/train.py, hier
    aber ohne Platzhalter für Stadt/Code – nur für den reinen
    Duplikat-Check bei Fragen gedacht.
    """
    normalized = text.strip().lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized


def question_exists(question_text):
    """Prüft per exaktem Abgleich der normalisierten Frage, ob sie schon
    in der Datenbank steht. Bewusst einfach gehalten (kein
    Embedding-Vergleich) – für eine robustere, semantische Prüfung könnte
    hier dieselbe sentence-transformers-Logik wie in retrieval.py
    verwendet werden (Ähnlichkeit statt exaktem Abgleich).
    """
    normalized = normalize_question(question_text)

    connection = get_connection()
    row = connection.execute(
        "SELECT id FROM questions WHERE normalized_text = ?",
        (normalized,),
    ).fetchone()
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
