import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
path = os.path.dirname(__file__)
# k = torch.load('project_1/data/k.pt',weights_only=False).numpy()
# args = np.where(k<0)
# print(args)
# plt.plot(np.sort(k,kind='heapsort'))
# plt.imshow(images[2222,0])
# plt.show()
# images_1 = torch.load('project_1/data/images_1.pt',weights_only=False)
# images_2 = torch.load('project_1/data/images_2.pt',weights_only=False)
# images_3 = torch.load('project_1/data/images_3.pt',weights_only=False)
# images_4 = torch.load('project_1/data/images_4.pt',weights_only=False)
# images_5 = torch.load('project_1/data/images_5.pt',weights_only=False)
# images_6 = torch.load('project_1/data/images_6.pt',weights_only=False)
# images = torch.cat((images_1,images_2,images_3,images_4,images_5,images_6),0)
# print(images.shape)

# k_1 = torch.load('project_1/data/k_1.pt',weights_only=False)
# k_2 = torch.load('project_1/data/k_2.pt',weights_only=False)
# k_3 = torch.load('project_1/data/k_3.pt',weights_only=False)
# k_4 = torch.load('project_1/data/k_4.pt',weights_only=False)
# k_5 = torch.load('project_1/data/k_5.pt',weights_only=False)
# k_6 = torch.load('project_1/data/k_6.pt',weights_only=False)
# k = torch.cat((k_1,k_2,k_3,k_4,k_5,k_6),0)
# print(k.shape)

# torch.save(images, os.path.join(path,'data/images.pt'))
# torch.save(k, os.path.join(path,'data/k.pt'))

# images = torch.load('project_1/data/images.pt',weights_only=False)
# model = torch.load(os.path.join(path,'models/cnn.pth'),weights_only=False)
# model.eval()
# print(model)
# first_conv = list(model.children())[0][0:1]
# img = torch.rand((1,1,128,128))
# img = images[0:1]
# print(model.conv_layers.conv1.weight.data.numpy().shape)
# output = first_conv(img).detach().numpy()
# print(output.shape)

# plt.figure()
# plt.imshow(img[0,0].numpy())

# plt.figure()
# for i in range(5):
#     plt.subplot(2,5,i+1)
#     plt.imshow(output[0,i])
#     plt.axis("off")
#     plt.subplot(2,5,i+6)
#     plt.imshow(model.conv_layers.conv1.weight.data.numpy()[i,0])
#     plt.axis("off")
# plt.tight_layout()
# plt.show()

# Load model and image
# images = torch.load('project_1/data/images.pt', weights_only=False)
# model = torch.load(os.path.join(path, 'models/cnn.pth'), weights_only=False)
# model.eval()

# # Select an input image
# img = images[0:1]  # Shape: (1, 1, 128, 128)

# # Function to extract convolutional layers
# def get_conv_layers(model):
#     conv_layers = []
#     for layer in model.children():
#         if isinstance(layer, nn.Sequential):  # Handle nested sequential layers
#             for sub_layer in layer:
#                 if isinstance(sub_layer, nn.Conv2d):
#                     conv_layers.append(sub_layer)
#         elif isinstance(layer, nn.Conv2d):
#             conv_layers.append(layer)
#     return conv_layers

# # Extract all convolutional layers
# conv_layers = get_conv_layers(model)

# # Pass image through each layer and visualize
# for idx, conv in enumerate(conv_layers):
#     with torch.no_grad():
#         img = conv(img)  # Apply convolution
#         output = img.numpy()

#     print(f"Layer {idx+1}: {conv}")
#     print(f"Output shape: {output.shape}")

#     # Plot original image or previous output
#     plt.figure(figsize=(10, 5))
#     for i in range(min(5, output.shape[1])):  # Limit to 5 feature maps
#         plt.subplot(2, 5, i + 1)
#         plt.imshow(output[0, i])
#         plt.axis("off")

#         plt.subplot(2, 5, i + 6)
#         plt.imshow(conv.weight.data.numpy()[i, 0])
#         plt.axis("off")

#     plt.suptitle(f"Layer {idx+1} Output & Filters")
#     plt.tight_layout()
#     plt.show()


# df_epoch_results = pd.read_csv(os.path.join(path,'training_data/cnn_best_long_01.csv'))

# sns.lineplot(df_epoch_results, x='Epoch',y='Train R2')
# sns.lineplot(df_epoch_results, x='Epoch',y='Test R2')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
