# 🤖 Local AI Assistant

A local AI assistant developed in Python consisting of two independent parts:

- **Part 1:** Local Chat Assistant using Ollama and Tkinter
- **Part 2:** Text Classification with DistilBERT (Hugging Face)

The project demonstrates both the use of a locally running Large Language Model (LLM) and the fine-tuning of a transformer model for text classification.

---

# 📂 Project Structure

```
Local-AI-Assistant/
│
├── teil1/
│   ├── gui.py
│   ├── ollama_client.py
│   └── ...
│
├── teil2/
│   ├── data/
│   │   └── dataset.csv
│   ├── train.py
│   ├── predict.py
│   ├── model/
│   └── ...
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Part 1 – Local AI Chat Assistant

## Goal

Develop a graphical desktop application that communicates with a locally running Large Language Model via Ollama.

## Technologies

- Python
- Tkinter
- Ollama
- Llama 3.1 / Mistral

## Features

- Local AI chatbot
- Graphical User Interface (GUI)
- Conversation history
- Communication via Ollama API

---

# 🧠 Part 2 – Text Classification with DistilBERT

## Goal

Fine-tune a pre-trained transformer model to classify customer messages into four categories.

## Categories

- Anfrage
- Reklamation
- Rechnung
- Sonstiges

---

## Dataset

The dataset was created manually and contains:

- **120 text examples**
- **4 categories**
- **30 examples per category**

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

# ⚙️ Training Process

The training pipeline consists of the following steps:

1. Load the dataset.
2. Convert category labels into numerical IDs.
3. Split the dataset into training and test sets.
4. Convert the data into Hugging Face Dataset format.
5. Tokenize the text using AutoTokenizer.
6. Fine-tune DistilBERT.
7. Evaluate the trained model.
8. Save the model and tokenizer.

---

# 📊 Training Results

| Parameter | Value |
|-----------|------|
| Model | distilbert-base-german-cased |
| Dataset size | 120 |
| Classes | 4 |
| Training samples | 96 |
| Test samples | 24 |
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Training time | ~0.57 min |
| Test Accuracy | 62.5% |

---

# 🔍 Prediction

After training, the saved model can classify new user messages.

Example:

Input:

```
Ich brauche meine Rechnung.
```

Output:

```
Kategorie: Rechnung
Confidence: 33.57%
```

---

# 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Local-AI-Assistant.git
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

### Part 1

Start Ollama:

```bash
ollama run llama3.1
```

Launch the GUI:

```bash
python teil1/gui.py
```

---

### Part 2

Train the model:

```bash
python teil2/train.py
```

Run prediction:

```bash
python teil2/predict.py
```

---

# 📚 Libraries

- transformers
- datasets
- torch
- pandas
- numpy
- scikit-learn
- tkinter
- requests

---

# 📌 Notes

This project was created for educational purposes to demonstrate:

- local LLM integration,
- transformer fine-tuning,
- natural language processing,
- text classification using Hugging Face.

Because the dataset is intentionally small (120 examples), the trained model may not correctly classify every unseen sentence. Increasing the dataset size and diversity would improve the overall accuracy.

---

# 📄 License

Educational project.