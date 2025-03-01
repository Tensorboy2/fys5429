import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, image_size=128, hidden_sizes=None):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [image_size * 8,
                            image_size * 8,
                            image_size * 2]
        
        layers = []
        input_dim = image_size * image_size
        
        # Construct hidden layers dynamically
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        B,C,Nx,Ny = x.shape
        x = self.layers(x.reshape(B,Nx*Ny))
        return x