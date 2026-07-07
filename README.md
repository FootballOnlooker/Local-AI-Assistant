# Local AI Chat Assistant

## Project Description

This project is Part 1 of the AQUA AI project.

It is a local AI chat assistant with a graphical user interface.  
The application was written in Python and uses Tkinter for the GUI.  
The AI model runs locally on the computer through Ollama.

The user can enter a message, send it to the local AI model and receive an answer directly in the chat window.

---

## Features

- Local AI chat assistant
- Python GUI with Tkinter
- Input field for user messages
- Send button
- Send message with Enter
- Chat history with scrollbar
- Automatic scrolling to the newest message
- Clear Chat button
- Error handling if Ollama or the model is not available
- Resizable window

---

## Technologies

- Python
- Tkinter
- Ollama
- Llama 3.2 3B

---

## Project Structure

```text
local-ai-chat-assistant/
│
├── main.py          # Starts the application
├── gui.py           # Graphical User Interface
├── chat.py          # Communication with Ollama
├── requirements.txt
├── README.md
│
└── screenshots/
    ├── app.png
    └── chat_history.png
```

---

## Installation

### 1. Install Ollama

Download and install Ollama:

```text
https://ollama.com
```

### 2. Download the AI model

```bash
ollama pull llama3.2:3b
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the application with:

```bash
python main.py
```

Ollama must be installed and available locally.

---

## Screenshots

### Main Application

![Local AI Chat Assistant](screenshots/app.png)

### Chat History

![Chat History](screenshots/chat_history.png)

---

## How It Works

1. The user writes a message in the input field.
2. The user clicks the Send button or presses Enter.
3. The GUI calls the function `send_message()`.
4. `send_message()` sends the text to `ask_ai()`.
5. `ask_ai()` communicates with the local Ollama model.
6. The AI answer is returned to the GUI.
7. The user message and AI answer are displayed in the chat history.

---

## Development Progress

### Part 1 – Local AI Chat Assistant

- [x] Install Ollama locally
- [x] Download local AI model
- [x] Test model in terminal
- [x] Create Python function for Ollama communication
- [x] Create Tkinter GUI
- [x] Add input field
- [x] Add Send button
- [x] Add chat history
- [x] Connect GUI with Ollama
- [x] Add Enter key support
- [x] Add Clear Chat button
- [x] Add error handling
- [x] Make window resizable
- [x] Add screenshots
- [x] Write README

### Part 2 – Small AI Training

- [ ] Create dataset with 100–200 examples
- [ ] Train a small pretrained model
- [ ] Evaluate the model
- [ ] Document training duration, examples, epochs and accuracy
- [ ] Optional: connect classifier with GUI

---

## Notes

This project uses a local AI model.  
No external API key is required.

The model is not trained in this part.  
The application only sends user messages to the locally running Ollama model and displays the response.
