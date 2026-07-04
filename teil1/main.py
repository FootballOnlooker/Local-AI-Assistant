import ollama


def ask_ai(prompt):
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


question = "Was ist Python?"

answer = ask_ai(question)

print("Frage:")
print(question)

print("\nAntwort:")
print(answer)