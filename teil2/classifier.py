import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FINAL_MODEL_PATH = BASE_DIR / "model" / "final_model"
tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL_PATH)

model.eval()


def classify_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)  # Получить вероятности
        confidence = torch.max(probabilities).item()  # Найти максимальную вероятность
        confidence = round(confidence * 100, 2)  # %
    ids = torch.argmax(outputs.logits, dim=-1).item()
    result = model.config.id2label[ids]
    return result, confidence
