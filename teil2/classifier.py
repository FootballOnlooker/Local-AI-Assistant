import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FINAL_MODEL_PATH = BASE_DIR / "model" / "final_model"
tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL_PATH)

model.eval()


def classify_text(text):
    """Classify German customer-service text and return label and confidence."""

    if not text.strip():
        raise ValueError("Text must not be empty.")
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)

    confidence_tensor, predicted_id_tensor = torch.max(
        probabilities,
        dim=-1,
    )

    predicted_id = predicted_id_tensor.item()
    confidence = round(confidence_tensor.item() * 100, 2)
    label = model.config.id2label[predicted_id]

    return label, confidence
