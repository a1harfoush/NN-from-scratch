import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 2)  # Input to hidden layer (2 inputs -> 2 neurons)
        self.output = nn.Linear(2, 1)  # Hidden to output layer (2 inputs -> 1 neuron)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.uniform_(self.hidden.weight, -0.5, 0.5)
        nn.init.uniform_(self.output.weight, -0.5, 0.5)
        nn.init.constant_(self.hidden.bias, 0.5)
        nn.init.constant_(self.output.bias, 0.7)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = torch.tanh(self.output(x))
        return x

model = SimpleNN()

input_tensor = torch.tensor([0.5, 0.5], dtype=torch.float32)

output = model(input_tensor)

print("Output of the neural network:", output.item())
