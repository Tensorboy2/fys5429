import torch 
import os
path = os.path.dirname(__file__)

'''
Code used to concat different data generation runs.
'''

images_1 = torch.load('project_1/data/images_1.pt',weights_only=False)
images_2 = torch.load('project_1/data/images_2.pt',weights_only=False)
images_3 = torch.load('project_1/data/images_3.pt',weights_only=False)
images_4 = torch.load('project_1/data/images_4.pt',weights_only=False)
images_5 = torch.load('project_1/data/images_5.pt',weights_only=False)
images_6 = torch.load('project_1/data/images_6.pt',weights_only=False)
images_7 = torch.load('project_1/data/images_7.pt',weights_only=False)
images_8 = torch.load('project_1/data/images_8.pt',weights_only=False)
images = torch.cat((images_1,images_2,images_3,images_4,images_5,images_6,images_7,images_8),0)
print(images.shape)

k_1 = torch.load('project_1/data/k_1.pt',weights_only=False)
k_2 = torch.load('project_1/data/k_2.pt',weights_only=False)
k_3 = torch.load('project_1/data/k_3.pt',weights_only=False)
k_4 = torch.load('project_1/data/k_4.pt',weights_only=False)
k_5 = torch.load('project_1/data/k_5.pt',weights_only=False)
k_6 = torch.load('project_1/data/k_6.pt',weights_only=False)
k_7 = torch.load('project_1/data/k_7.pt',weights_only=False)
k_8 = torch.load('project_1/data/k_8.pt',weights_only=False)
k = torch.cat((k_1,k_2,k_3,k_4,k_5,k_6,k_7,k_8),0)
print(k.shape)

torch.save(images, os.path.join(path,'data/images.pt'))
torch.save(k, os.path.join(path,'data/k.pt'))


