from pathlib import Path
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

# Kalibriert mit drei Testfragen: passende Frage nah am Wortlaut ~0.69,
# relevante aber umformulierte Frage ~0.41, thematisch irrelevante Frage
# ~0.26. 0.35 liegt sicher über dem irrelevanten und unter beiden
# relevanten Fällen.
MIN_SIMILARITY = 0.35

# Referenz-Codes (z. B. "EXP-70531", "RE-2024-1187") sind kurze, seltene
# Tokens, die Embeddings nicht zuverlässig erkennen — ein
# Substring-Abgleich ist hier zuverlässiger als semantische Ähnlichkeit.
REFERENCE_CODE_PATTERN = re.compile(r"\b[A-ZÄÖÜ]{2,5}-[0-9][0-9A-Z-]{2,}\b")

# Einmal beim Modul-Import geladen, nicht bei jeder Anfrage — sonst würde
# das Modell bei jeder Chat-Nachricht neu von der Festplatte geladen.
MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def extract_documented_cities(documents):
    """Liest alle Städte aus den 'Wien nach X: ...'-Zeilen von
    lieferung.txt, damit die Liste nicht manuell gepflegt werden muss."""
    cities = set()
    route_pattern = re.compile(r"Wien nach ([A-ZÄÖÜ][a-zäöüß]+)")

    for document in documents:
        if document["name"] == "lieferung.txt":
            cities.update(route_pattern.findall(document["text"]))

    return cities


def find_undocumented_delivery_city(question, documented_cities):
    """Deterministische Prüfung (kein LLM): Wird nach einer Stadt
    gefragt, die NICHT in lieferung.txt dokumentiert ist, gibt diese
    Funktion den Stadtnamen zurück, sonst None. Einfache Heuristik
    (Stadtname nach "nach"), kein vollständiger NLP-Ansatz.
    """
    mentioned_cities = re.findall(r"nach ([A-ZÄÖÜ][a-zäöüß]+)", question)

    for city in mentioned_cities:
        if city not in documented_cities and city != "Wien":
            return city

    return None


def extract_delivery_times(documents):
    """Wie extract_documented_cities(), zusätzlich mit der jeweils
    dokumentierten Lieferzeit als Text. Gegenstück zu
    find_undocumented_delivery_city(): liefert die exakte Angabe für
    Städte, die SEHR WOHL dokumentiert sind.
    """
    delivery_times = {}
    route_pattern = re.compile(r"Wien nach ([A-ZÄÖÜ][a-zäöüß]+):\s*(.+)")

    for document in documents:
        if document["name"] == "lieferung.txt":
            for city, description in route_pattern.findall(document["text"]):
                delivery_times[city] = description.strip()

    return delivery_times


def find_mentioned_documented_city(text, delivery_times):
    """Sucht im übergebenen Text nach einer bekannten Stadt (ganzes
    Wort). Bewusst für den GESAMTEN Gesprächsverlauf gedacht, nicht nur
    die aktuelle Frage — ein reiner Textabgleich verwässert (anders als
    eine Embedding-Suche) nicht dadurch, dass man ihm mehr Text gibt.
    """
    for city, description in delivery_times.items():
        if re.search(rf"\b{re.escape(city)}\b", text):
            return city, description

    return None, None


def load_documents():
    """Load all text files from the knowledge directory."""

    documents = []

    for file_path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append(
                {
                    "name": file_path.name,
                    "text": text,
                }
            )

    return documents


def chunk_documents(documents, min_chunk_words=15):
    """Teilt jedes Dokument in Absätze auf, statt es als Ganzes zu
    encodieren. Grund: Das Embedding-Modell hat max_seq_length=128
    Tokens; längere Dokumente wurden dahinter abgeschnitten und waren
    für die Suche unsichtbar. Kurze Absätze werden mit dem nächsten
    zusammengeführt. Für die Antwort ans LLM wird trotzdem immer der
    VOLLSTÄNDIGE Dokumenttext verwendet (full_text) — nur die Suche
    arbeitet auf Chunk-Ebene.
    """
    chunks = []

    for document in documents:
        paragraphs = [p.strip() for p in document["text"].split("\n\n") if p.strip()]
        buffer = ""

        for paragraph in paragraphs:
            buffer = f"{buffer} {paragraph}".strip() if buffer else paragraph

            if len(buffer.split()) >= min_chunk_words:
                chunks.append(
                    {
                        "name": document["name"],
                        "full_text": document["text"],
                        "chunk_text": buffer,
                    }
                )
                buffer = ""

        if buffer:
            chunks.append(
                {
                    "name": document["name"],
                    "full_text": document["text"],
                    "chunk_text": buffer,
                }
            )

    return chunks


def encode_chunks(chunks):
    """Berechnet Embeddings für Chunks (Absätze) statt ganzer Dokumente."""
    if not chunks:
        return np.empty((0, MODEL.get_embedding_dimension()))

    chunk_texts = [chunk["chunk_text"] for chunk in chunks]
    return MODEL.encode(chunk_texts)


def find_document_by_reference_code(question, documents):
    """Exakter Fallback für Referenz-Codes, vor der Embedding-Suche
    geprüft. Gibt alle Dokumente zurück, die einen gefundenen Code
    enthalten — ein Code kann bewusst in mehreren Dokumenten vorkommen.
    """
    codes = REFERENCE_CODE_PATTERN.findall(question.upper())
    if not codes:
        return []

    found_documents = []
    for document in documents:
        document_text_upper = document["text"].upper()
        for code in codes:
            if code in document_text_upper:
                found_documents.append(document)
                break

    return found_documents


def retrieve_document(question, documents, chunks=None, chunk_vectors=None):
    """Findet das relevanteste Dokument für eine Frage: zuerst exakter
    Referenz-Code-Abgleich, sonst Embedding-Suche auf Chunk-Ebene (siehe
    chunk_documents()). Gibt trotzdem den vollständigen Dokumenttext
    zurück, damit das LLM den vollen Kontext bekommt.
    """
    if not question.strip():
        return {
            "name": None,
            "text": "",
            "similarity": 0.0,
        }

    if not documents:
        return {
            "name": None,
            "text": "",
            "similarity": 0.0,
        }

    exact_matches = find_document_by_reference_code(question, documents)
    if exact_matches:
        combined_name = ", ".join(document["name"] for document in exact_matches)
        combined_text = "\n\n---\n\n".join(document["text"] for document in exact_matches)
        return {
            "name": combined_name,
            "text": combined_text,
            "similarity": 1.0,
            "exact_match": True,
        }

    if chunks is None:
        # Fallback für manuelles Testen ohne vorberechnete Chunks.
        chunks = chunk_documents(documents)

    if not chunks:
        return {
            "name": None,
            "text": "",
            "similarity": 0.0,
        }

    if chunk_vectors is None:
        chunk_vectors = encode_chunks(chunks)

    question_vector = MODEL.encode([question])
    similarities = cosine_similarity(
        question_vector,
        chunk_vectors,
    )[0]

    best_index = int(np.argmax(similarities))
    best_similarity = float(similarities[best_index])

    if best_similarity < MIN_SIMILARITY:
        return {
            "name": None,
            "text": "",
            "similarity": best_similarity,
        }

    best_chunk = chunks[best_index]
    return {
        "name": best_chunk["name"],
        "text": best_chunk["full_text"],
        "similarity": best_similarity,
    }


if __name__ == "__main__":
    loaded_documents = load_documents()
    loaded_chunks = chunk_documents(loaded_documents)
    loaded_chunk_vectors = encode_chunks(loaded_chunks)

    question = input("Enter your question: ").strip()
    result = retrieve_document(
        question,
        loaded_documents,
        loaded_chunks,
        loaded_chunk_vectors,
    )

    print("\nRetrieved document:")
    print(f"Name: {result['name']}")
    print("Similarity:", f"{result['similarity']:.4f}")
    print("\nText:")
    print(result["text"])
