from classifier import classify_text

text = input("Enter text: ").strip()

result, confidence = classify_text(text)

if not text:
    print("Bitte geben Sie einen Text ein.")
else:
    print(f"Kategorie: {result}")
    print(f"Confidence: {confidence} %")
