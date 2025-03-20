'''A module containing a pytorch class for doing Convolution Neural Network processing of 2d images with 1 channel to single parameter.'''
import torch
import torch.nn as nn
torch.manual_seed(0)
class CNN(nn.Module):
    def __init__(self,
                 image_size=128,
                 # Default convolution layout:
                 conv_layers_params=[{'out_channels': 5, 'kernel_size': 3, 'stride': 2, 'pool': 'max'},
                                     {'out_channels': 10, 'kernel_size': 3, 'stride': 1, 'pool': 'max'},
                                     {'out_channels': 20, 'kernel_size': 3, 'stride': 1, 'pool': 'max'}],
                 hidden_sizes=[1,1],
                 activation='relu',
                 use_dropout=False,
                 use_batch_norm=False):
        """
        Parameters:
            image_size (int): The size (width/height) of the input image.
            conv_layers_params (list): List of dictionaries, each with:
                - out_channels (int): Number of output channels.
                - kernel_size (int): Size of the convolution kernel.
                - stride (int): Stride for the convolution.
                - pool (str or None): Optional pooling method ('max' or 'avg').
            hidden_sizes (list): List of sizes for fully connected hidden layers.
            activation (str): Activation function to use ('relu', 'leakyrelu', 'sigmoid', 'tanh').
            use_dropout (bool): Whether to include dropout (p=0.5) after fully connected layers.
            use_batch_norm (bool): Whether to add batch normalization after each conv layer.
        """
        super().__init__()

        
        def get_activation(name): # Getting the right activation function
            if name == 'relu':
                return nn.ReLU()
            elif name == 'leakyrelu':
                return nn.LeakyReLU()
            elif name == 'sigmoid':
                return nn.Sigmoid()
            elif name == 'tanh':
                return nn.Tanh()
            else:
                raise ValueError(f"Unsupported activation: {name}")

        # Build convolutional layers:
        self.conv_layers = nn.Sequential()
        in_channels = 1  # A single-channel input from binary image
        current_size = image_size
        pool_kernel = 2  # Default pooling kernel size
        pool_stride = 2  # Default pooling stride

        for idx, params in enumerate(conv_layers_params):
            conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=params['out_channels'],
                             kernel_size=params['kernel_size'],
                             stride=params['stride'])
            
            self.conv_layers.add_module(f"conv{idx+1}", conv)

            current_size = (current_size - params['kernel_size']) // params['stride'] + 1 # Output size after convolution
            
            if use_batch_norm: # Optional batch normalization
                bn = nn.BatchNorm2d(params['out_channels'])
                self.conv_layers.add_module(f"batch_norm{idx+1}", bn)

            self.conv_layers.add_module(f"activation{idx+1}", get_activation(activation)) # Activation

            
            if params.get('pool', None) is not None: # Optional pooling layer
                if params['pool'] == 'max':
                    pool_layer = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
                
                elif params['pool'] == 'avg':
                    pool_layer = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
                
                else: # Handel invalid pooling type input
                    raise ValueError("Pooling must be 'max', 'avg', or None")
                self.conv_layers.add_module(f"pool{idx+1}", pool_layer)
                
                current_size = (current_size - pool_kernel) // pool_stride + 1 # Output after pooling:

            in_channels = params['out_channels']

        
        conv_output_dim = in_channels * (current_size ** 2) # Input size to fully connected layers.

        linear = []
        for hidden in hidden_sizes:
            linear.append(conv_output_dim // hidden)

        # Build fully connected layers:
        fc_layers = []
        fc_input_dim = conv_output_dim
        for idx, hidden_dim in enumerate(linear):
            fc_layers.append(nn.Linear(fc_input_dim, hidden_dim))

            if use_dropout: # Apply dropout if given.
                fc_layers.append(nn.Dropout(p=0.5))
            fc_layers.append(get_activation(activation))
            fc_input_dim = hidden_dim

        fc_layers.append(nn.Linear(fc_input_dim, 1)) # Output layer
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Forward pass:
          x: Tensor of shape (B, 1, image_size, image_size). Single channel image.
        Returns:
          out: Tensor with the predicted parameter.
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers (B,conv_output_dim*conv_output_dim)
        out = self.fc_layers(x)
        return out


if __name__ == '__main__':
    import torch
    image_size = 128
    x = torch.rand((1,1,image_size,image_size)) # (B,C,Nx,Ny) We only have one channel but torch still expects a channel number.
    model = CNN(use_dropout=True, use_batch_norm=True)
    pred = model(x).item()
    print(pred)
