from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
MIN_SIMILARITY = 0.05


def retrieve_documents(question, documents):
    print(f'Question: {question}')
    document_text = [document['text'] for document in documents]
    vectorizer = TfidfVectorizer(lowercase=True)
    document_vectors = vectorizer.fit_transform(document_text)
    question_vector = vectorizer.transform([question])

    feature_names = vectorizer.get_feature_names_out()
    similarities = cosine_similarity(
        question_vector,
        document_vectors,
    )[0]

    best_index = np.argmax(similarities)
    best_document = documents[best_index]
    best_similarity = similarities[best_index]

    if best_similarity < MIN_SIMILARITY:
        return {
            "name": None,
            "text": "",
            "similarity": best_similarity,
        }

    for word_index, weight in zip(
            question_vector.indices,
            question_vector.data,
    ):
        word = feature_names[word_index]
        print(f"{word}: {weight:.4f}")
    print(vectorizer)


def load_documents():
    """Load all text files from the knowledge directory."""

    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append(
                {
                    "name": file_path.name,
                    "text": text,
                }
            )

    return documents


if __name__ == "__main__":
    loaded_documents = load_documents()

    question = input("Enter your question: ").strip()
    result = retrieve_documents(
        question,
        loaded_documents,
    )

    print("\nRetrieved document:")
    print(f"Name: {result["name"]}")
    print("Similarity:", f"{result['similarity']:.4f}")
    print("\nText:")
    print(result["text"])
