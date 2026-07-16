![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-Educational-green)

# 🤖 Local AI Assistant

A local AI assistant developed in Python that combines a local Large Language Model (LLM) with a fine-tuned transformer model for text classification.

The project consists of two connected parts:

- **Part 1:** Local AI Chat Assistant using Ollama and Tkinter.
- **Part 2:** German customer message classification using DistilBERT.

The classifier is integrated into the GUI and automatically predicts the category of every user message before the AI generates its response.

---

# 📂 Project Structure

```text
Local-AI-Assistant/
│
├── teil1/
│   ├── chat.py
│   ├── gui.py
│   └── main.py
│
├── teil2/
│   ├── data/
│   │   └── dataset_teil2.csv
│   ├── model/
│   │   └── final_model/
│   ├── check_dataset.py
│   ├── classifier.py
│   ├── predict.py
│   └── train.py
│
├── images/
│   ├── teil1_chat.png
│   └── training_model_teil2.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🚀 Part 1 – Local AI Chat Assistant

## Goal

Develop a desktop AI assistant that communicates with a locally running Large Language Model using Ollama.

## Technologies

- Python
- Tkinter
- Ollama
- Llama 3.2:3b

## Features

- Local AI chatbot
- Graphical User Interface (GUI)
- Conversation history
- Communication with Ollama
- Automatic text classification
- Category prediction
- Confidence score display

---

# 🧠 Part 2 – Text Classification with DistilBERT

## Goal

Fine-tune a transformer model to classify German customer messages into predefined categories.

## Categories

- Anfrage
- Reklamation
- Rechnung
- Sonstiges

---

## Dataset

The dataset was manually created.

It contains:

- **200 text examples**
- **4 categories**
- **50 examples per category**

The dataset is stored as a CSV file.

---

## Technologies

- Python
- Pandas
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-Learn

---

# ⚙️ Training Pipeline

The training process consists of the following steps:

1. Load the dataset.
2. Encode category labels into numerical IDs.
3. Split the dataset into training and test sets.
4. Convert the data into Hugging Face Dataset format.
5. Tokenize the text.
6. Fine-tune DistilBERT.
7. Evaluate the model.
8. Save the trained model.
9. Save the tokenizer.

---

# 📊 Training Results

| Parameter | Value |
|-----------|------|
| Model | distilbert-base-german-cased |
| Dataset size | 200 |
| Classes | 4 |
| Training samples | 160 |
| Test samples | 40 |
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Training time | ~0.82 min |
| Evaluation Loss | 0.8794 |
| Test Accuracy | **87.50 %** |

---

# 🔍 Prediction Example

Input

```text
Ich habe meine Rechnung nicht erhalten.
```

Output

```text
Kategorie: Rechnung

Confidence: 87.50 %
```

---

# 🖥 GUI Example

Every user message is automatically classified before the AI generates a response.

Example

```text
Sie:
Ich habe meine Rechnung nicht erhalten.

Kategorie:
Rechnung (87.50 %)

KI:
Natürlich helfe ich Ihnen.
Bitte senden Sie mir Ihre Bestellnummer oder Rechnungsnummer,
damit ich Ihnen weiterhelfen kann.
```

---

# 💻 Requirements

- Python 3.10+
- Ollama installed
- Llama 3.2:3b downloaded locally

---

# 🛠 Installation

Clone the repository

```bash
git clone https://github.com/FootballOnlooker/Local-AI-Assistant.git
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## 1. Download the Ollama model

```bash
ollama pull llama3.2:3b
```

---

## 2. Train the classifier

```bash
python teil2/train.py
```

The training process creates

```text
teil2/model/final_model/
```

---

## 3. Run the GUI

```bash
python teil1/main.py
```

---

## 4. Optional standalone prediction

```bash
python teil2/predict.py
```

---

# 📚 Main Libraries

- transformers
- datasets
- torch
- pandas
- numpy
- scikit-learn
- ollama
- tkinter

---

# 📸 Screenshots

## Local AI Chat Assistant

![Local AI Chat Assistant](images/teil1_chat.png)

---

## DistilBERT Training

![DistilBERT Training](images/training_model_teil2.png)

---

# 📌 Notes

This project was created for educational purposes.

It demonstrates:

- Local LLM integration using Ollama
- Transformer fine-tuning with Hugging Face
- Text classification using DistilBERT
- Desktop GUI development with Tkinter
- Integration of a classifier into a local AI assistant

The classifier is trained on a manually created dataset containing **200 German customer messages**.

Although the achieved accuracy is **87.50%**, the model may still produce incorrect predictions for previously unseen or ambiguous messages.

The displayed confidence score is the highest Softmax probability produced by the classifier. A high confidence value indicates that the model strongly prefers one category over the others, but it does **not** guarantee that the prediction is correct.

---

# 📄 Usage

This project was created for educational and demonstration purposes.