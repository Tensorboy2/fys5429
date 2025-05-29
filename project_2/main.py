import torch
import torch.nn as nn
torch.manual_seed(0)
import torch.optim as optim
import time
import os
path = os.path.dirname(__file__)


from train import train
from models.resnet import ResNet50, ResNet101, ResNet152
from models.convnext import ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge
from models.vit import ViT_B16, ViT_B4, ViT_B8, ViT_L16, ViT_H16
from data_loader import get_data

model_registry = {
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "ConvNeXtTiny": ConvNeXtTiny,
    "ConvNeXtSmall": ConvNeXtSmall,
    "ConvNeXtBase": ConvNeXtBase,
    "ConvNeXtLarge": ConvNeXtLarge,
    "ViT_B16": ViT_B16,
    "ViT_B4": ViT_B4,
    "ViT_B8": ViT_B8,
    "ViT_L16": ViT_L16,
    "ViT_H16": ViT_H16
}

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def main(model, hyperparameters, data, save_path="metrics.csv"):
    num_epochs = hyperparameters["num_epochs"]
    lr = hyperparameters["lr"]
    warmup_steps = hyperparameters["warmup_steps"]
    weight_decay = hyperparameters["weight_decay"]
    batch_size = hyperparameters["batch_size"]
    decay = hyperparameters["decay"]

    torch.cuda.empty_cache() # Make available space
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    optimizer = optim.AdamW(params = model.parameters(),
                             lr = lr, 
                             weight_decay=weight_decay)
    model.to(device)
    print(model.name)
    train_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                   test_size=data["test_size"],
                                                   rotate=data["rotate"],
                                                   hflip=data["hflip"],
                                                   vflip=data["vflip"],
                                                   group=data["group"],
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
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load a YAML config file.")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # result_dir = config["result_dir"]
    for exp in config["experiments"]:
        model_name = exp["model"]
        model_class = model_registry[model_name]
        model = model_class()
        print(f"Model size: {get_model_size(model):.2f} MB")

        main(model, exp["hyperparameters"], exp["data"], save_path=exp["save_path"])

        del model
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
