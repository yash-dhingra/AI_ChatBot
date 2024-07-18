import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

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
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if tag=="component_issues":
        # Ask if the user is comfortable with handeling hardware
        print(f"{bot_name}  Tag({tag}): I'm really sorry to hear that you're experiencing this issue.\n If you're comfortable with handling PC hardware, we could try some potential solutions together.?(Yes or No)")
        inp=input("You: ")
        if inp=="Yes" or inp=="yes" or inp=="YES" or inp=="y" or inp=="Y":
            print(f"{bot_name}  Tag({tag}): Great! Let's start by opening the case and checking the connections of the components inside.\n Refer this Image: https://www.google.com/url?sa=i&url=https%3A%2F%2Fin.pinterest.com%2Fpin%2F410742428485532076%2F&psig=AOvVaw0eLukg3XQiX6SQCVAw4RlR&ust=1721244191435000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCJDBurykrIcDFQAAAAAdAAAAABAT")
        else:
            print(f"{bot_name}  Tag({tag}): No worries! Let's Connect you to an executive.")


        # Ask if it resolved the issue
        print(f"{bot_name}  Tag({tag}): Did this resolve the issue?(Yes or No)")
        inp=input("You: ")
        if inp=="Yes" or inp=="yes" or inp=="YES" or inp=="y" or inp=="Y":
            print(f"{bot_name}  Tag({tag}): Great! I'm glad I could help you. Is there anything else I can assist you with?")
        else:
            print(f"{bot_name}  Tag({tag}): No worries! Let's Connect you to an executive.")

    elif tag=="windows_not_installed":
        # Ask if the user is comfortable with installing windows
        print(f"{bot_name}  Tag({tag}): I'm really sorry to hear that you're experiencing this issue.\n If you're comfortable with installing Windows, we could try some potential solutions together.?(Yes or No)")
        inp=input("You: ")
        if inp=="Yes":
            print(f"{bot_name}  Tag({tag}): Great! Let's start by downloading the Windows 10 ISO file from the official Microsoft website.\n Refer this link: https://www.microsoft.com/en-in/software-download/windows10")
        else:
            print(f"{bot_name}  Tag({tag}): No worries! Let's Connect you to an executive.")


        # Ask if it resolved the issue
        print(f"{bot_name}  Tag({tag}): Did this resolve the issue?(Yes or No)")
        inp=input("You: ")
        if inp=="Yes":
            print(f"{bot_name}  Tag({tag}): Great! I'm glad I could help you. Is there anything else I can assist you with?")
        else:
            print(f"{bot_name}  Tag({tag}): No worries! Let's Connect you to an executive.")



    elif tag=="check_warranty":
        # Ask user for orderID and check the order.csv file for checking warranty
        print(f"{bot_name}  Tag({tag}): I'm really sorry to hear that you're experiencing this issue.\n Could you please provide me with the Order ID?")
        inp=input("You: ")
        print(f"{bot_name}  Tag({tag}): Let me check the warranty status for you.")
        # Check the order.csv file for the warranty status
        # TO BE DONE LATER
        print(f"{bot_name}  Tag({tag}): Did this resolve the issue?(Yes or No)")
        inp=input("You: ")
        if inp=="Yes":
            print(f"{bot_name}  Tag({tag}): Great! I'm glad I could help you. Is there anything else I can assist you with?")
        else:
            print(f"{bot_name}  Tag({tag}): No worries! Let's Connect you to an executive.")
    
    elif tag=="noisy_fan" or tag=="damaged_cpu_cabinet_received" or tag=="components_seems_missing" or tag=="cant_register_for_warranty" or tag=="need_help_for_software_setup" :
        # Connect with an executive
        print(f"{bot_name}  Tag({tag}): I'm really sorry to hear that you're experiencing this issue.\n Let's Connect you to an executive.")
        break

            
    else:
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print(f"{bot_name}  Tag({tag}): {random.choice(intent['responses'])}")
                    # print(f"{bot_name}  Tag({tag}): Did this resolve the issue?(Yes or No)")
                    # inp=input("You: ")
                    # if inp=="Yes" or inp=="yes" or inp=="YES" or inp=="y" or inp=="Y":
                    #     print(f"{bot_name}  Tag({tag}): Great! I'm glad I could help you. Is there anything else I can assist you with?")
                    # else:
                    #     print(f"{bot_name}  Tag({tag}): No worries! Let's Connect you to an executive.")
                            
        else:
            print(f"{bot_name}: I do not understand...")