import tkinter as tk
from tkinter import scrolledtext
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import pandas as pd
from datetime import datetime, timedelta

# Load the CSV file
df = pd.read_csv('dummy_orders.csv')

def check_warranty(order_id, bot_name, tag):
    # Find the order in the dataframe
    order = df[df['OrderID'] == order_id]
    
    if order.empty:
        return f"Order ID {order_id} not found."
    
    # Get the date of purchase
    date_of_purchase = datetime.strptime(order.iloc[0]['DateOfPurchase'], '%Y-%m-%d')
    
    # Calculate the expiry date
    expiry_date = date_of_purchase + timedelta(days=365)
    
    # Check if the warranty is still valid
    if datetime.now() <= expiry_date:
        return f"{bot_name} Tag({tag}): The warranty for Order ID {order_id} is still valid.\nExpiry Date: {expiry_date.strftime('%Y-%m-%d')}"
    else:
        return f"{bot_name} Tag({tag}): The warranty for Order ID {order_id} has expired.\nExpiry Date: {expiry_date.strftime('%Y-%m-%d')}"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "ViShaYa"
awaiting_order_id = False
awaiting_hardware_issue_response = False
awaiting_windows_installation_response = False
awaiting_hardware_issue_follow_up = False
awaiting_windows_installation_follow_up = False

def respond_to_user(user_input):
    global awaiting_order_id, awaiting_hardware_issue_response, awaiting_windows_installation_response
    global awaiting_hardware_issue_follow_up, awaiting_windows_installation_follow_up

    if awaiting_order_id:
        awaiting_order_id = False
        return check_warranty(user_input, bot_name, "check_warranty")

    if awaiting_hardware_issue_response:
        awaiting_hardware_issue_response = False
        if user_input.lower() in ["yes", "y"]:
            awaiting_hardware_issue_follow_up = True
            return f"{bot_name} Tag(component_issues): Great! Let's start by opening the case and checking the connections of the components inside.\nRefer to this Image: https://www.google.com/url?sa=i&url=https%3A%2F%2Fin.pinterest.com%2Fpin%2F410742428485532076%2F&psig=AOvVaw0eLukg3XQiX6SQCVAw4RlR&ust=1721244191435000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCJDBurykrIcDFQAAAAAdAAAAABAT\n\nDid this resolve the issue? (Yes or No)"
        else:
            return f"{bot_name} Tag(component_issues): No worries! Let's connect you to an executive."

    if awaiting_hardware_issue_follow_up:
        awaiting_hardware_issue_follow_up = False
        if user_input.lower() in ["yes", "y"]:
            return f"{bot_name} Tag(component_issues): Great! I'm glad I could help you. Is there anything else I can assist you with?"
        else:
            return f"{bot_name} Tag(component_issues): No worries! Let's connect you to an executive."

    if awaiting_windows_installation_response:
        awaiting_windows_installation_response = False
        if user_input.lower() in ["yes", "y"]:
            awaiting_windows_installation_follow_up = True
            return f"{bot_name} Tag(windows_not_installed): Great! Let's start by downloading the Windows 10 ISO file from the official Microsoft website.\nRefer to this link: https://www.microsoft.com/en-in/software-download/windows10\n\nDid this resolve the issue? (Yes or No)"
        else:
            return f"{bot_name} Tag(windows_not_installed): No worries! Let's connect you to an executive."

    if awaiting_windows_installation_follow_up:
        awaiting_windows_installation_follow_up = False
        if user_input.lower() in ["yes", "y"]:
            return f"{bot_name} Tag(windows_not_installed): Great! I'm glad I could help you. Is there anything else I can assist you with?"
        else:
            return f"{bot_name} Tag(windows_not_installed): No worries! Let's connect you to an executive."

    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    response = ""
    if tag == "component_issues":
        awaiting_hardware_issue_response = True
        response = f"{bot_name} Tag({tag}): I'm really sorry to hear that you're experiencing this issue.\nIf you're comfortable with handling PC hardware, we could try some potential solutions together. (Yes or No)"
    elif tag == "windows_not_installed":
        awaiting_windows_installation_response = True
        response = f"{bot_name} Tag({tag}): I'm really sorry to hear that you're experiencing this issue.\nIf you're comfortable with installing Windows, we could try some potential solutions together. (Yes or No)"
    elif tag == "check_warranty":
        awaiting_order_id = True
        response = f"{bot_name} Tag({tag}): I'm really sorry to hear that you're experiencing this issue.\n\nCould you please provide me with the Order ID?"
    elif tag in ["noisy_fan", "damaged_cpu_cabinet_received", "components_seems_missing", "cant_register_for_warranty", "need_help_for_software_setup"]:
        response = f"{bot_name} Tag({tag}): I'm really sorry to hear that you're experiencing this issue.\n\nLet's connect you to an executive."
    else:
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = f"{bot_name} Tag({tag}): {random.choice(intent['responses'])}"
        else:
            response = f"{bot_name}: I do not understand..."

    response += "\n"
    return response

def send():
    user_input = user_entry.get()
    chat_window.insert(tk.END, "You: " + user_input + "\n\n")
    
    if user_input.lower() in ["quit", "exit", "bye", "goodbye", "no"]:
        chat_window.insert(tk.END, f"{bot_name}: Goodbye!\n")
        root.quit()
    else:
        response = respond_to_user(user_input)
        chat_window.insert(tk.END, response + "\n")
        chat_window.yview(tk.END)
        user_entry.delete(0, tk.END)

# Creating GUI with Tkinter
root = tk.Tk()
root.title("ViShaYa Chatbot")

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20, font=("Arial", 10))
chat_window.pack(padx=10, pady=10)

user_entry = tk.Entry(root, width=50)
user_entry.pack(padx=10, pady=10)
user_entry.bind("<Return>", lambda event: send())

send_button = tk.Button(root, text="Send", command=send)
send_button.pack(padx=10, pady=10)

root.mainloop()
