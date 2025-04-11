from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import os
path = os.path.dirname(__file__)


def get_data(batch_size = 32,test_size=0.2,normalize=True, mask=False, grid_search=None, device=None):
    '''
    Function for getting data and turning them into the train and test loader.
    Optional: normalization, grid search size, mask outliers.
    '''
    if not None: # Less data for grid search
        images = torch.load(os.path.join(path,'data/images.pt'),weights_only=False)[:grid_search] # Load generated images
        params = torch.load(os.path.join(path,'data/k.pt'),weights_only=False)[:grid_search] # Load calculated permeability
    else:
        images = torch.load(os.path.join(path,'data/images.pt'),weights_only=False) # Load generated images
        params = torch.load(os.path.join(path,'data/k.pt'),weights_only=False) # Load calculated permeability

    
    if mask: # Mask outliers
        mask = (params > 0) #& (params <= 0.05)
        images, params = images[mask], params[mask]

    if normalize: # Normalize
        params_mean, params_std = params.mean(), params.std()
        params = (params-params_mean)/params_std

    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(images.to(device), params.to(device), test_size=test_size,random_state=42)
    train_dataset = TensorDataset(X_train,y_train)
    test_dataset = TensorDataset(X_test,y_test)
    train_data_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset,batch_size=batch_size)
    return train_data_loader, test_data_loader