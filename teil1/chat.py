import ollama
from retrieval import load_documents, retrieve_document

DOCUMENTS = load_documents()


def ask_ai(question):
    retrieval_result = retrieve_document(
        question,
        DOCUMENTS,
    )
    context = retrieval_result["text"]

    if not context:
        return (
            "Diese Information befindet sich nicht "
            "in den bereitgestellten Dokumenten."
        )

    prompt = f"""
    Du bist ein hilfreicher Kundenservice-Assistent.

    Beantworte die Frage ausschließlich anhand des bereitgestellten Kontexts.
    Erfinde keine zusätzlichen Informationen.

    Kontext:
    {context}

    Frage:
    {question}

    Antwort:
    """.strip()

    try:
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"]

    except Exception as error:
        return (
            "Die KI-Antwort konnte nicht erstellt werden. "
            "Bitte prüfen Sie, ob Ollama gestartet ist."
        )
