#Ich importiere die Bibliothek ollama, damit mein Python-Programm mit dem lokalen KI-Modell kommunizieren kann.
import ollama

# Send a user message to the local Ollama model and return the AI response.
    response = ollama.chat(
        model="llama3.2:3b", #wie heißt model Ollama
        messages=[
            {
                "role": "user", #Die Rolle ist „user“, weil diese Nachricht vom Benutzer kommt.
                "content": prompt
            }
        ]
    )

    # Return only the generated text from the response.
    return response["message"]["content"] 
