from pathlib import Path
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

# Bei TF-IDF lag der Schwellenwert bei 0.05, weil die Ähnlichkeit ohne
# gemeinsame Wörter praktisch bei 0 liegt. Embeddings haben ein höheres
# "Grundrauschen" (~0.2-0.3 auch bei thematisch unpassenden Sätzen auf
# derselben Sprache), daher muss der Schwellenwert deutlich höher liegen.
# Kalibrierung mit drei echten Testfragen (empirisch, nicht geraten):
#   - "Wien nach Sofia" (Frage nah am Dokumentwortlaut):      similarity 0.69
#   - "Rechnung nicht erhalten" (Frage, umformuliert):        similarity 0.41
#   - "Wie ist das Wetter" (thematisch irrelevant):            similarity 0.26
# 0.55 hätte den zweiten (echt relevanten) Fall verworfen. 0.35 liegt mit
# Sicherheitsabstand über dem irrelevanten Fall (0.26) und unter beiden
# relevanten Fällen (0.41, 0.69).
MIN_SIMILARITY = 0.35

# Referenz-Codes (Sendungs-, Rechnungs-, Policennummern, z. B. "EXP-70531",
# "RE-2024-1187") sind kurze, seltene alphanumerische Tokens. Embeddings
# tun sich damit erfahrungsgemäß schwer (siehe Diskussion zu
# "Standardverkehrart": seltene Fachbegriffe/Codes ohne breiten
# Trainingskontext werden vom Modell nicht scharf im Vektorraum platziert).
# Für exakte Codes ist ein einfacher Substring-Abgleich zuverlässiger als
# semantische Ähnlichkeit.
REFERENCE_CODE_PATTERN = re.compile(r"\b[A-ZÄÖÜ]{2,5}-[0-9][0-9A-Z-]{2,}\b")

# Wird EINMAL beim Import des Moduls geladen, genau wie tokenizer/model in
# teil2/classifier.py. Würde diese Zeile stattdessen in retrieve_document()
# stehen, würde bei jeder einzelnen Chat-Nachricht das komplette Modell neu
# von der Festplatte geladen (daher die wiederholten "Loading weights"-Logs).
MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def extract_documented_cities(documents):
    """Liest alle Städte aus den 'Wien nach X: ...'-Zeilen von
    lieferung.txt. Dynamisch aus der Datei extrahiert, damit die Liste
    nicht von Hand im Code gepflegt werden muss, wenn jemand später
    weitere Strecken in lieferung.txt ergänzt.
    """
    cities = set()
    route_pattern = re.compile(r"Wien nach ([A-ZÄÖÜ][a-zäöüß]+)")

    for document in documents:
        if document["name"] == "lieferung.txt":
            cities.update(route_pattern.findall(document["text"]))

    return cities


def find_undocumented_delivery_city(question, documented_cities):
    """Deterministische Prüfung (kein LLM-Aufruf): Wird nach der Lieferzeit
    zu einer Stadt gefragt, die NICHT in lieferung.txt dokumentiert ist,
    gibt diese Funktion den Stadtnamen zurück. Sonst None.

    Hinweis/Trade-off: Das ist ein einfacher Heuristik-Check (Städtenamen
    nach "nach"), kein vollständiger NLP-Ansatz. Er deckt genau den in den
    Tests aufgetretenen Fall ab (z. B. "Transport nach Klagenfurt"), ist
    aber nicht so allgemein wie die Referenz-Code-Prüfung oben.
    """
    mentioned_cities = re.findall(r"nach ([A-ZÄÖÜ][a-zäöüß]+)", question)

    for city in mentioned_cities:
        if city not in documented_cities and city != "Wien":
            return city

    return None


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
    """Teilt jedes Dokument in Absätze für die Embedding-Suche auf.

    Hintergrund (empirisch bestätigt): Das Embedding-Modell hat ein
    max_seq_length von 128 Tokens. Mehrere Wissensdokumente sind als
    ganzer Text länger als das (z. B. 170 Wörter -> geschätzt 220+ Token
    durch deutsche Subword-Tokenisierung). Alles nach der 128-Token-Grenze
    wurde beim Encodieren des GANZEN Dokuments schlicht abgeschnitten und
    war für die Suche unsichtbar — unabhängig davon, wie wichtig der Inhalt
    dort war (z. B. ein neu ergänzter Absatz am Ende einer Datei).

    Lösung: Nicht das ganze Dokument auf einmal encodieren, sondern in
    kürzere Absätze aufteilen und JEDEN separat encodieren. Sehr kurze
    Absätze (z. B. nur eine Überschrift) werden mit dem nächsten
    zusammengeführt, damit jeder Chunk genug Kontext für ein
    aussagekräftiges Embedding hat. Für die Antwort an das LLM wird
    trotzdem immer der VOLLSTÄNDIGE Dokumenttext verwendet (full_text) —
    die Aufteilung betrifft nur die Suche, nicht den Kontext, den das
    Modell am Ende sieht.
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
    """Exakter Fallback für Referenz-Codes (Sendungs-, Rechnungs-,
    Policennummern). Wird vor der Embedding-Suche geprüft, weil kurze
    alphanumerische Codes vom Embedding-Modell nicht zuverlässig erkannt
    werden (siehe MIN_SIMILARITY-Kommentar oben).

    Gibt eine Liste aller Dokumente zurück, die einen gefundenen Code
    enthalten (leere Liste, wenn kein Code im Text vorkommt oder keiner
    davon in einem Dokument gefunden wurde). Ein Code kann absichtlich in
    mehreren Dokumenten vorkommen (z. B. EXP-70531 sowohl in kontakt.txt
    als Beispiel als auch in rechnung.txt als eigentliche Antwort) – daher
    wird hier nicht beim ersten Treffer abgebrochen.
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
    """Return the most relevant knowledge document for a question.

    Sucht auf Ebene von Absätzen (chunks), nicht ganzen Dokumenten, um die
    128-Token-Grenze des Embedding-Modells zu umgehen (siehe
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
        # Fallback, falls jemand die Funktion ohne vorberechnete Chunks
        # aufruft (z. B. beim manuellen Testen über die Kommandozeile).
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
