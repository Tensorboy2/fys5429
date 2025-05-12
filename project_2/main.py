import torch
import torch.nn as nn
torch.manual_seed(0)
import torch.optim as optim
import time
import os
path = os.path.dirname(__file__)


from train import train
from models.resnet import ResNet50, ResNet101, ResNet50V2
# from models.convnext import ConvNeXtTiny, ConvNeXtSmall, ConvNeXtXL
from models.vit import ViT_B16, ViT_L16, ViT_H16
from data_loader import get_data

model_registry = {
    "ResNet50": ResNet50,
    "ResNet50V2": ResNet50V2,
    "ResNet101": ResNet101,
    "ViT_B16": ViT_B16,
    "ViT_L16": ViT_L16,
    "ViT_H16": ViT_H16
}


def main(model, hyperparameters, data, save_path="metrics.csv"):
    num_epochs = hyperparameters["num_epochs"]
    lr = hyperparameters["lr"]
    warmup_steps = hyperparameters["warmup_steps"]
    weight_decay = hyperparameters["weight_decay"]
    batch_size = hyperparameters["batch_size"]
    decay = hyperparameters["decay"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    optimizer = optim.AdamW(params = model.parameters(),
                             lr = lr, 
                             weight_decay=weight_decay)
    model.to(device)
    print(model)
    train_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                   test_size=data["test_size"],
                                                   normalize=data["normalize"],
                                                   num_samples=data["num_samples"])
    start = time.time()
    train(model,
            optimizer,
            train_data_loader,
            test_data_loader, 
            num_epochs=num_epochs,
            warmup_steps=warmup_steps,
            decay=decay,
            save_path=save_path,
            device=device)
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')


import yaml
if __name__ == '__main__':
    with open("configs/vit_b16.yaml", "r") as f:
        config = yaml.safe_load(f)

    for exp in config["experiments"]:
        model_name = exp["model"]
        model_class = model_registry[model_name]
        model = model_class()
        main(model, exp["hyperparameters"], exp["data"], save_path=exp["save_path"])
    
