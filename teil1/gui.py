from tkinter import *
from tkinter.scrolledtext import ScrolledText
from chat import ask_ai

from teil2.classifier import classify_text


def send_message():
    user_text = user_input.get()

    if user_text.strip() == '':
        return
    ai_answer = ask_ai(user_text)

    result, confidence = classify_text(user_text)

    chat_history.config(state=NORMAL)

    chat_history.insert(END, "-" * 50 + "\n\n")

    chat_history.insert(END, "Sie:\n")
    chat_history.insert(END, f"{user_text}\n\n")

    chat_history.insert(END, "Kategorie:\n")
    chat_history.insert(
        END,
        f"{result} ({confidence:.2f} %)\n\n"
    )

    chat_history.insert(END, "KI:\n")
    chat_history.insert(END, f"{ai_answer}\n\n")

    chat_history.insert(END, "-" * 50 + "\n\n")
    chat_history.see(END)

    chat_history.config(state=DISABLED)

    user_input.delete(0, END)  # Clean user text


def clear_chat():
    chat_history.config(state=NORMAL)
    chat_history.delete("1.0", END)
    chat_history.config(state=DISABLED)


window = Tk()
window.title('Local AI Chat Assistant')
window.config(padx=30, pady=20)

title_label = Label(
    text='Local AI Chat Assistant',
    font=("Arial", 16, "bold")
)
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

chat_history = ScrolledText(
    width=70,
    height=20,
    font=("Arial", 11),
    wrap=WORD
)

chat_history.grid(row=1, column=0, columnspan=2, sticky="nsew")
chat_history.config(state=DISABLED)
window.grid_rowconfigure(1, weight=1)
window.grid_columnconfigure(0, weight=1)

user_input = Entry(
    width=50,
    font=("Arial", 12)
)
user_input.grid(row=2, column=0, pady=12, sticky="ew")


def paste_text(event=None):
    try:
        clipboard_text = window.clipboard_get()
        user_input.insert(INSERT, clipboard_text)
    except TclError:
        pass
    return "break"


user_input.bind("<Control-v>", paste_text)
user_input.bind("<Control-V>", paste_text)
send_button = Button(
    text="Senden",
    width=12,
    command=send_message
)
window.bind("<Return>", lambda event: send_message())
send_button.grid(row=2, column=1, padx=10)

clear_button = Button(
    text="Clear Chat",
    width=12,
    command=clear_chat
)

clear_button.grid(row=3, column=0, columnspan=2, pady=5)
window.minsize(700, 500)


def run_app():
    window.mainloop()
