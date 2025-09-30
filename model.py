import torch  # Import PyTorch library
import torch.nn as nn  # Import neural network module from PyTorch


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()  # Initialize the parent nn.Module
        self.l1 = nn.Linear(
            input_size, hidden_size
        )  # First linear layer (input to hidden)
        self.l2 = nn.Linear(
            hidden_size, hidden_size
        )  # Second linear layer (hidden to hidden)
        self.l3 = nn.Linear(
            hidden_size, num_classes
        )  # Third linear layer (hidden to output)
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        out = self.l1(x)  # Pass input through first layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.l2(out)  # Pass through second layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.l3(out)  # Pass through third layer to get output
        # no activation and no softmax at the end (handled externally)
        return out  # Return raw output logits
