import random  # Import random for selecting random responses
import json  # Import json for loading intents from a file
import torch  # Import torch for PyTorch model operations
from model import NeuralNet  # Import the custom neural network model
from nltk_utils import bag_of_words, tokenize  # Import NLP utility functions

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Set device to GPU if available, else CPU

with open("love_intents.json", "r") as f:  # Open and read the intents JSON file
    intents = json.load(f)  # Load intents data from the file

FILE = "love_chatbot_model.pth"  # Path to the trained model file
data = torch.load(FILE)  # Load the model data from file

input_size = data["input_size"]  # Get input size for the model
hidden_size = data["hidden_size"]  # Get hidden layer size
output_size = data["output_size"]  # Get output size for the model
all_words = data["all_words"]  # Get list of all words used in training
tags = data["tags"]  # Get list of tags (intents)
model_state = data["model_state"]  # Get the trained model state dictionary

model = NeuralNet(input_size, hidden_size, output_size).to(
    device
)  # Initialize the neural network model
model.load_state_dict(model_state)  # Load the trained weights into the model
model.eval()  # Set the model to evaluation mode

bot_name = "LoveBot"  # Name of the chatbot


def get_response(msg):
    sentence = tokenize(msg)  # Tokenize the user's message
    X = bag_of_words(sentence, all_words)  # Convert tokens to bag-of-words vector
    X = X.reshape(1, X.shape[0])  # Reshape for model input
    X = torch.from_numpy(X).to(
        device
    )  # Convert numpy array to torch tensor and move to device

    output = model(X)  # Get model predictions
    _, predicted = torch.max(
        output, dim=1
    )  # Get the index of the highest predicted score

    tag = tags[predicted.item()]  # Get the tag corresponding to the prediction

    probs = torch.softmax(output, dim=1)  # Apply softmax to get probabilities
    prob = probs[0][predicted.item()]  # Get probability of the predicted tag
    if prob.item() > 0.75:  # If confidence is high enough
        for intent in intents["intents"]:  # Search for the matching intent
            if tag == intent["tag"]:  # If tag matches intent
                return random.choice(
                    intent["responses"]
                )  # Return a random response from intent

    return "I do not understand..."  # Default response if confidence is low
