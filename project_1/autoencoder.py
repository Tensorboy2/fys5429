import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self,image_size=128,hidden_sizes= [256,128], depth = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = nn.Linear(image_size*image_size,hidden_sizes[0])
        self.hidden_layer_1 = nn.Linear(hidden_sizes[0],hidden_sizes[1])
        self.output_layer = nn.Linear(hidden_sizes[1],1)
        self.layers = nn.Sequential(self.input_layer,
                                    nn.ReLU(),
                                    self.hidden_layer_1,
                                    nn.ReLU(),
                                    self.output_layer)

    def forward(self,x):
        B,C,Nx,Ny = x.shape
        x = self.layers(x.reshape(B,Nx*Ny))
        return x