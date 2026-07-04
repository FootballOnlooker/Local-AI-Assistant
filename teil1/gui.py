from tkinter import *
from tkinter.scrolledtext import ScrolledText
from main import ask_ai

def send_message():
    user_text = user_input.get()

    if user_text.strip() == '':
        return
    ai_answer = ask_ai(user_text)

    chat_history.config(state=NORMAL)
    chat_history.insert(END, f'You: {user_text}\n')
    chat_history.insert(END, f'AI: {ai_answer}\n\n')

    chat_history.config(state=DISABLED)

    user_input.delete(0, END) # Clean user text


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

chat_history.grid(row=1, column=0, columnspan=2)
chat_history.config(state=DISABLED)


user_input = Entry(
    width=50,
    font=("Arial", 12)
)
user_input.grid(row=2, column=0, pady=12, sticky="w")

send_button = Button(
    text="Senden",
    width=12,
    command=send_message
)

send_button.grid(row=2, column=1, padx=10)

window.mainloop()




