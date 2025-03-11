import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, image_size=128, hidden_sizes=None):
        super().__init__()
        
        
        layers = []
        input_dim = image_size * image_size

        if hidden_sizes is None:
            hidden_sizes = [input_dim // 16,
                            input_dim // 32]
        
        # Construct hidden layers dynamically:
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Dropout(p=0.5))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        B,C,Nx,Ny = x.shape
        x = self.layers(x.reshape(B,Nx*Ny))
        return x
    

if __name__ == '__main__':
    x = torch.rand((1,1,128,128))
    model = Autoencoder()
    pred = model(x).item()
    print(pred)