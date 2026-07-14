import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "teil2/model/final_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()
text = input("Enter text: ").strip()


def classify_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)  # Получить вероятности
        confidence = torch.max(probabilities).item()  # Найти максимальную вероятность
        confidence = round(confidence * 100, 2) # %
    ids = torch.argmax(outputs.logits, dim=-1).item()
    result = model.config.id2label[ids]
    return result, confidence


result, confidence = classify_text(text)

if not text:
    print("Bitte geben Sie einen Text ein.")
else:
    print(f"Kategorie: {result}")
    print(f"Confidence: {confidence} %")
