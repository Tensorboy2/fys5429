import torch.nn as nn
import torch
torch.manual_seed(0)
class FeedForward(nn.Module):
    def __init__(self, image_size=128, hidden_sizes=[2, 4], activation='relu'):
        super().__init__()

        def get_activation(name):
            activations = {
                'relu': nn.ReLU(),
                'leakyrelu': nn.LeakyReLU(),
                'sigmoid': nn.Sigmoid(),
                'tanh': nn.Tanh()
            }
            if name not in activations:
                raise ValueError(f"Unsupported activation: {name}")
            return activations[name]

        layers = []
        input_dim = image_size * image_size

        # Construct hidden layers dynamically
        for hidden in hidden_sizes:
            hidden_dim = input_dim // hidden
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Dropout(p=0.5))
            layers.append(get_activation(activation))
            input_dim = hidden_dim  # Update for next layer

        # Output layer
        layers.append(nn.Linear(input_dim, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        B, C, Nx, Ny = x.shape
        x = x.view(B, Nx * Ny)  # Flatten input
        return self.layers(x)
    

if __name__ == '__main__':
    x = torch.rand((1,1,128,128))
    model = FeedForward()
    pred = model(x).item()
    print(pred)