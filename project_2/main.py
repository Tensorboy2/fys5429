import torch
import torch.nn as nn
torch.manual_seed(0)
import torch.optim as optim
import time
import os
path = os.path.dirname(__file__)


from train import train
from models.resnet import ResNet50, ResNet101, ResNet50V2
from models.convnext import ConvNeXtTiny, ConvNeXtSmall, ConvNeXtXL
from models.vit import ViT_B16, ViT_L16, ViT_H16

model_registry = {
    "ResNet50": ResNet50,
    "ResNet50V2": ResNet50V2,
    "ResNet101": ResNet101,
    "ConvNeXtTiny": ConvNeXtTiny,
    "ConvNeXtSmall": ConvNeXtSmall,
    "ConvNeXtXL": ConvNeXtXL,
    "ViT_B16": ViT_B16,
    "ViT_L16": ViT_L16,
    "ViT_H16": ViT_H16
}

from data_loader import get_data

def main(model, hyperparameters, data, save_path="metrics.csv"):
    num_epochs = hyperparameters["num_epochs"]
    lr = hyperparameters["lr"]
    lr_step = hyperparameters["lr_step"]
    weight_decay = hyperparameters["weight_decay"]
    batch_size = hyperparameters["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    optimizer = optim.AdamW(params = model.parameters(),
                             lr = lr, 
                             weight_decay=weight_decay)
    model.to(device)
    print(model)
    train_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                   test_size=0.2,
                                                   normalize=data["normalize"],
                                                   num_samples=data["num_samples"],
                                                   device=device)
    start = time.time()
    train(model,
            optimizer,
            train_data_loader,
            test_data_loader, 
            num_epochs=num_epochs,
            lr_step = lr_step,
            save_path=save_path)
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')


import yaml
if __name__ == '__main__':
    with open("configs/resnet50_vs_resnet50v2.yaml", "r") as f:
        config = yaml.safe_load(f)

    for exp in config["experiments"]:
        model_name = exp["model"]
        model_class = model_registry[model_name]
        model = model_class()
        main(model, exp["hyperparameters"], exp["data"], save_path=exp["save_path"])

    # model = ResNet50()
    # hyperparameters = {"num_epochs":1000,
    #                 "lr":0.001,
    #                 "lr_step":1000,
    #                 "weight_decay":1e-2,
    #                 "batch_size":32}
    # data = {"normalize":True,
    #         "mask":True,
    #         "grid_search":None}
    # main(model, hyperparameters, data,save_path=f"resnet50_1000_epochs_metrics.csv")
    
    # model = ResNet101()
    # hyperparameters = {"num_epochs":1000,
    #                 "lr":0.001,
    #                 "lr_step":1000,
    #                 "weight_decay":1e-2,
    #                 "batch_size":32}
    # data = {"normalize":True,
    #         "mask":True,
    #         "grid_search":None}
    # main(model, hyperparameters, data,save_path=f"resnet101_1000_epochs_metrics.csv")

    """
    Test different data set sizes:
    """
    # for i in range(1000,9001,1000):
    #     print(i)
    #     model = ResNet50()
    #     hyperparameters = {"num_epochs":100,
    #                     "lr":0.001,
    #                     "lr_step":200,
    #                     "weight_decay":1e-2,
    #                     "batch_size":32}
    #     data = {"normalize":True,
    #             "mask":True,
    #             "grid_search":i}
    #     main(model, hyperparameters, data,save_path=f"resnet50_{i}.csv")

        # model = ConvNeXtTiny()
        # hyperparameters = {"num_epochs":200,
        #                 "lr":0.0001,
        #                 "lr_step":200,
        #                 "weight_decay":1e-5,
        #                 "batch_size":32}
        # data = {"normalize":True,
        #         "mask":True,
        #         "grid_search":i}
        # main(model, hyperparameters, data)
    
