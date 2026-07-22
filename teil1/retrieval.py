from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
MIN_SIMILARITY = 0.12


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


def retrieve_document(question, documents):
    """Return the most relevant knowledge document for a question."""
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

    document_texts = [document['text'] for document in documents]
    vectorizer = TfidfVectorizer(lowercase=True)
    document_vectors = vectorizer.fit_transform(document_texts)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(
        question_vector,
        document_vectors,
    )[0]

    best_index = int(np.argmax(similarities))
    best_similarity = float(similarities[best_index])

    if best_similarity < MIN_SIMILARITY:
        return {
            "name": None,
            "text": "",
            "similarity": best_similarity,
        }
    best_document = documents[best_index]
    return {
        "name": best_document["name"],
        "text": best_document["text"],
        "similarity": best_similarity,
    }


if __name__ == "__main__":
    loaded_documents = load_documents()

    question = input("Enter your question: ").strip()
    result = retrieve_document(
        question,
        loaded_documents,
    )

    print("\nRetrieved document:")
    print(f"Name: {result['name']}")
    print("Similarity:", f"{result['similarity']:.4f}")
    print("\nText:")
    print(result["text"])
