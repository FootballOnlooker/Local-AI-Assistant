from tkinter import *
from tkinter.scrolledtext import ScrolledText
from main import ask_ai

# This function is executed when the user clicks the "Senden" button.
def send_message():
    # Get the text from the input field.
    user_text = user_input.get()

     # If the input is empty or only spaces, do nothing.
    if user_text.strip() == '':
        return
     # Send the user's message to the local AI model and get the answer.
    ai_answer = ask_ai(user_text)

     # Enable the chat history field so the program can write into it.
    chat_history.config(state=NORMAL)
    # Add the user's message to the chat history.
    chat_history.insert(END, f'You: {user_text}\n')
    # Add the AI answer to the chat history.
    chat_history.insert(END, f'AI: {ai_answer}\n\n')
    
     # Disable the chat history again so the user cannot edit it.
    chat_history.config(state=DISABLED)

     # Clear the input field after sending the message.
    user_input.delete(0, END) # Clean user text

# Create the main application window.
window = Tk()
# Set the title of the window.
window.title('Local AI Chat Assistant')
# Add space around all widgets inside the window.
window.config(padx=30, pady=20)

# Create the title label at the top of the window.
title_label = Label(
    text='Local AI Chat Assistant',
    font=("Arial", 16, "bold")
)
# Place the title in row 0.
# columnspan=2 means: the title uses two columns.
# pady=(0, 10) means: no space above, 10 pixels space below.
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

# Create the chat history field with a scrollbar.
chat_history = ScrolledText(
    width=70,
    height=20,
    font=("Arial", 11),
    wrap=WORD
)
# Place the chat history in row 1.
# It also uses two columns.
chat_history.grid(row=1, column=0, columnspan=2)
# Disable the chat history so the user cannot edit it manually.
chat_history.config(state=DISABLED)

# Create the input field where the user writes a message.
user_input = Entry(
    width=50,
    font=("Arial", 12)
)
# Place the input field in row 2, column 0.
# sticky="w" means: align it to the left side.
user_input.grid(row=2, column=0, pady=12, sticky="w")

# Create the send button.
send_button = Button(
    text="Senden",
    width=12,
    command=send_message
)
# Place the button in row 2, column 1.
# padx=10 adds horizontal space around the button.
send_button.grid(row=2, column=1, padx=10)

# Start the Tkinter event loop.
# The window stays open and waits for user actions.
window.mainloop()




