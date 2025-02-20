'''A module containing a pytorch class for doing Convolution Neural Network processing of 2d images with 1 channel to single parameter.'''
import torch.nn as nn

class CNN(nn.Module):
    '''
    A python class for a CNN using pytorch.
    '''
    def __init__(self, image_size = 128, 
                 num_out_channels_1 = 32,
                 kernel_size_1 = 3,
                 stride_1 = 1,
                 pool_1 = None,
                 num_out_channels_2 = 64,
                 kernel_size_2 = 3,
                 stride_2 = 1,
                 pool_2 = None,
                 hidden_size=128,
                 activation = 'relu',
                 use_dropout = False,
                 use_batch_norm = False,
                    *args, **kwargs):
        '''
        # Parameters:
        - image_size: Int (The size of the input image)
        - num_out_channels_1: Int (The number of produced feature maps, similar number of kernels)
        - kernel_size_1: Int (The size of the filter (kernel))
        - stride_1: Int (The numbers of steps between each convolution)
        - pool_1: 'max' or 'avg' (Optional. Sets the pooling method used)
        - num_out_channels_2: Int (The number of produced feature maps, similar number of kernels)
        - kernel_size_2: Int (The size of the filter (kernel))
        - stride_2: Int (The numbers of steps between each convolution)
        - pool_2: 'max' or 'avg' (Optional. Sets the pooling method used)
        - hidden_size: Int (The number of hidden nodes in the fully connected layer)
        - activation: 'relu', 'leakyrelu', 'sigmoid' or 'tanh' (Activation function in fully connected layer)
        - use_dropout: Bool (Optional. Whether or not to use dropout)
        - use_batch_norm: Bool (Optional. Whether or not to use batch normalization)

        The constructor of a CNN need parameters for its basic architecture.
        In this project we will stick to only having 2 layers of convolution,
        but letting the kernel size, stride and channels be tunable hyper parameters.
        For the fully connected layer we will let its architecture be fixed to 2 hidden 
        layers but the with will be tunable. For the model we will assume that the image is 
        square so that the width and hight is the same number of pixels.

        The model will assume that the batch (B) comes first then the channel (C) then the dimension of the image (B,C,Nx,Ny).
        '''
        super().__init__(*args, **kwargs)

        self.training = False # Can be used for later if tests if needed. (DELETE IF NEVER USED IN THE PROJECT)

        # Declaring the first convolution layer.
        self.convolution_1 = nn.Conv2d(in_channels = 1, # Only one channel since data is binary.
                                       out_channels = num_out_channels_1, # Same as number of kernels.
                                       kernel_size = kernel_size_1,
                                       stride = stride_1)
        
        # Declaring the second convolution layer.
        self.convolution_2 = nn.Conv2d(in_channels = num_out_channels_1, # Must match the output of the last layer.
                                       out_channels = num_out_channels_2,
                                       kernel_size = kernel_size_2,
                                       stride = stride_2)
        
        self.activation_cnn = nn.ReLU()


        # Declaring optional pooling layers.
        self.pool_stride = 2
        self.pool_kernel = 2
        self.pool_1 = pool_1
        self.pool_2 = pool_2
        if pool_1 == 'max':
            self.pool_1 = nn.MaxPool2d(kernel_size = self.pool_kernel, stride = self.pool_stride)
        elif pool_1 == 'avg':
            self.pool_1 = nn.AvgPool2d(kernel_size = self.pool_kernel, stride = self.pool_stride)

        if pool_2 == 'max':
            self.pool_2 = nn.MaxPool2d(kernel_size = self.pool_kernel, stride = self.pool_stride)
        elif pool_2 == 'avg':
            self.pool_2 = nn.AvgPool2d(kernel_size = self.pool_kernel, stride = self.pool_stride)

        '''
        To account for the size of the out channels of the convolution and possibly pooling layers we calculate:
        H = ((((image_size - kernel_size) / conv_stride) + 1) - (pool_kernel - 1))/ pool_stride
        '''
        # Compute feature map size after first convolution
        out_conv_1 = ((image_size - kernel_size_1) // stride_1) + 1
        if pool_1 is not None:
            out_conv_1 = ((out_conv_1 - self.pool_kernel) // self.pool_stride) + 1
        # Compute feature map size after second convolution
        out_conv_2 = ((out_conv_1 - kernel_size_2) // stride_2) + 1
        if pool_2 is not None:
            out_conv_2 = ((out_conv_2 - self.pool_kernel) // self.pool_stride) + 1 


        # Declare fully connected layer with correct input size
        self.first_linear_layer = nn.Linear(in_features = num_out_channels_2 * (out_conv_2 ** 2),
                                             out_features = hidden_size)
        
        # Declare activation function in fully connected layers
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        # Declare last layer 
        self.second_linear_layer = nn.Linear(in_features = hidden_size,
                                             out_features = 1)
        
        # Declare optional dropout
        '''
        Dropout can be used in many places of a neural network. It can be effectual
        in parts that are prone to over fitting. One such part is the first linear layer 
        as it has a big number of input parameters compared to other parts of the network.
        '''
        self.use_dropout = use_dropout
        if use_dropout:
            self.p = 0.5 # Making p an attribute so that it can be reset later
            self.dropout = nn.Dropout(p = self.p) # 0.5 is the default value, currently do not know of a better value 

        # Declare optional batch normalization
        '''
        Convolution layers can be regularized by the use of batch normalization.  
        '''
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_1 = nn.BatchNorm2d(num_out_channels_1)
            self.batch_norm_2 = nn.BatchNorm2d(num_out_channels_2)

    def forward(self,x):
        '''
        Applies the CNN to a input image and return a number.

        # Params:
        - x: tensor (The image of form (B,Nx,Ny))

        # Returns:
        - out: tensor (The predicted parameter)
        '''
        x = self.convolution_1(x) # Apply first convolution layer (B,C,Nx,Ny)
        x = self.activation_cnn(x) # Apply ReLU to feature maps
        if self.pool_1 is not None: # Apply pooling if given
            x = self.pool_1(x)
        if self.use_batch_norm: # Apply batch normalization if given
            x = self.batch_norm_1(x)
        x = self.convolution_2(x) # Apply second convolution layer
        x = self.activation_cnn(x)# Apply ReLU to feature maps
        if self.pool_2 is not None: # Apply pooling if given
            x = self.pool_2(x)
        if self.use_batch_norm: # Apply batch normalization if given
            x = self.batch_norm_2(x)
        x = x.view(x.shape[0], -1) # Flatten to give to the linear layers
        x = self.first_linear_layer(x) # Apply first linear layer
        x = self.activation(x) # Apply activation function
        if self.use_dropout: # Apply dropout if given
            x = self.dropout(x) 
        out = self.second_linear_layer(x) # Apply second linear layer
        return out


if __name__ == '__main__':
    import torch
    image_size = 128
    x = torch.rand((1,1,image_size,image_size)) # (B,C,Nx,Ny) We only have one channel but torch still expects a channel number.
    model = CNN(image_size=image_size, pool_1='max',pool_2='avg', activation='sigmoid', use_dropout=True, use_batch_norm=True)
    r = model(x).item()
    print(r)
