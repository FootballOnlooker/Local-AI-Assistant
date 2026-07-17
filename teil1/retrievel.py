from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"


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

    print(f"Loaded documents: {len(loaded_documents)}")

    for document in loaded_documents:
        print(f"- {document['name']}")