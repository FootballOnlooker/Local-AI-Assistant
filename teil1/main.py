import ollama


def ask_ai(prompt):
    try:
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]
    except Exception as e:
        return f"Fehler: Ollama ist nicht erreichbar. Details: {e}"
