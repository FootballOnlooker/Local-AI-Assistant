import time
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

BASE_DIR = Path(__file__).resolve().parent
FINAL_MODEL_DIR = BASE_DIR / "model" / "final_model"
start_time = time.time()

# 1. Load dataset
DATASET_PATH = BASE_DIR / "data" / "dataset_teil2.csv"

df = pd.read_csv(DATASET_PATH, sep=';')

print("Dataset loaded:")
print(df.head())
print(df["label"].value_counts())

# 2. Labels to numbers
label2id = {
    "Anfrage": 0,
    "Reklamation": 1,
    "Rechnung": 2,
    "Sonstiges": 3,
}

id2label = {
    0: "Anfrage",
    1: "Reklamation",
    2: "Rechnung",
    3: "Sonstiges",
}

df["label_id"] = df["label"].map(label2id)

# 3. Train/Test split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"]
)

# 4. Convert pandas to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 5. Load tokenizer and model
model_name = "distilbert-base-german-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    id2label=id2label,
    label2id=label2id
)


# 6. Tokenization
def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Trainer expects column name "labels"
train_dataset = train_dataset.rename_column("label_id", "labels")
test_dataset = test_dataset.rename_column("label_id", "labels")

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(["text", "label", "__index_level_0__"])
test_dataset = test_dataset.remove_columns(["text", "label", "__index_level_0__"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")


# 7. Accuracy function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


# 8. Training settings
training_args = TrainingArguments(
    output_dir="model",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="teil2/logs",
    logging_steps=10,
    load_best_model_at_end=False,
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 10. Start training
trainer.train()

# 11. Final evaluation
results = trainer.evaluate()

# 12. Save model and tokenizer
trainer.save_model(str(FINAL_MODEL_DIR))
tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

end_time = time.time()
training_minutes = round((end_time - start_time) / 60, 2)
print("Training Summary")
print("=" * 40)

print(f"Model: {model_name}")
print(f"Dataset size: {len(df)}")
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Training time: {training_minutes:.2f} min")
print(f"Evaluation Loss: {results['eval_loss']:.4f}")
print(f"Test Accuracy: {results['eval_accuracy'] * 100:.2f}%")
print("Model saved to: teil2/model/final_model")

print("=" * 40)
