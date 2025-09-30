import json  # Import JSON for reading intents file
import numpy as np  # Import NumPy for numerical operations
import torch  # Import PyTorch for tensor operations
import torch.nn as nn  # Import neural network module from PyTorch
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Import Dataset and DataLoader for batching
from nltk_utils import tokenize, stem, bag_of_words  # Import NLP utility functions
from model import NeuralNet  # Import the neural network model

with open("love_intents.json", "r") as f:  # Load intents from JSON file
    intents = json.load(f)

all_words = []  # List to store all words from patterns
tags = []  # List to store all unique tags
xy = []  # List to store (tokenized pattern, tag) pairs
for intent in intents["intents"]:  # Iterate through each intent
    tag = intent["tag"]  # Get the tag for the intent
    tags.append(tag)  # Add tag to tags list
    for pattern in intent["patterns"]:  # Iterate through each pattern
        w = tokenize(pattern)  # Tokenize the pattern
        all_words.extend(w)  # Add tokens to all_words list
        xy.append((w, tag))  # Store (tokens, tag) pair

ignore_words = ["?", "!", ".", ","]  # Punctuation to ignore
all_words = [
    stem(w) for w in all_words if w not in ignore_words
]  # Stem and filter words
all_words = sorted(set(all_words))  # Remove duplicates and sort
tags = sorted(set(tags))  # Remove duplicate tags and sort

# Create training data
X_train = []  # List for input features
y_train = []  # List for target labels
for pattern_sentence, tag in xy:  # Iterate through each (tokens, tag) pair
    bag = bag_of_words(
        pattern_sentence, all_words
    )  # Convert tokens to bag-of-words vector
    X_train.append(bag)  # Add feature vector to X_train

    label = tags.index(tag)  # Get index of tag for label
    y_train.append(label)  # Add label to y_train

X_train = np.array(X_train)  # Convert training features to NumPy array
y_train = np.array(y_train)  # Convert training labels to NumPy array


class ChatDataset(Dataset):  # Custom dataset for chat data
    def __init__(self):
        self.n_samples = len(X_train)  # Number of samples
        self.x_data = X_train  # Feature data
        self.y_data = y_train  # Label data

    def __getitem__(self, index):  # Get item by index
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # Get total number of samples
        return self.n_samples


# Hyperparameters
batch_size = 8  # Number of samples per batch
num_workers = 7  # Number of worker threads (not used, set to 0 below)
hidden_size = 8  # Number of neurons in hidden layers
output_size = len(tags)  # Number of output classes
input_size = len(X_train[0])  # Size of input feature vector
learning_rate = 0.001  # Learning rate for optimizer
num_epochs = 1000  # Number of training epochs

dataset = ChatDataset()  # Create dataset instance
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)  # Create data loader for batching

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Set device to GPU if available
model = NeuralNet(
    input_size, hidden_size, output_size
)  # Initialize neural network model
model.to(device)  # Move model to device


# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Train the model
for epoch in range(num_epochs):  # Loop over epochs
    for words, labels in train_loader:  # Loop over batches
        words = words.to(device).float()  # Move input to device and convert to float
        labels = labels.to(device).long()  # Move labels to device and convert to long

        outputs = model(words)  # Forward pass through model
        loss = criterion(outputs, labels)  # Compute loss

        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")  # Print final loss after training

data = {
    "model_state": model.state_dict(),  # Trained model weights
    "input_size": input_size,  # Input size
    "output_size": output_size,  # Output size
    "hidden_size": hidden_size,  # Hidden layer size
    "all_words": all_words,  # Vocabulary
    "tags": tags,  # List of tags
}

FILE = "love_chatbot_model.pth"  # File to save trained model
torch.save(data, FILE)  # Save model and metadata
print(f"Training complete. File saved to {FILE}")  # Print completion message
