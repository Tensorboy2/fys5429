'''Training module for CNN.'''
import torch
import torch.optim as op
import numpy as np
from sklearn.metrics import r2_score
import csv

@torch.jit.script
def r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Jit'ed rd score function
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def train(model = None, optimizer = None, train_data_loader = None, test_data_loader = None, num_epochs = 5, lr_step = 5,save_path="metrics.csv"):
    '''
    Training function.
    Performs adams optimization to model on train data loader and predicts on test data loader.
    Returns metrics from training.
    '''
    metrics = {
        "epoch": [],
        "train_mse" : [], # Mean square error
        "train_r2" : [], # Coefficient of Determination
        "test_mse" : [], # Mean square error
        "test_r2" : [] # Coefficient of Determination
    }

    loss_fn = torch.nn.MSELoss()
    scheduler = op.lr_scheduler.StepLR(optimizer,step_size=lr_step)

    for epoch in range(num_epochs):
        running_train_loss = 0
        all_y_true_train = []
        all_y_pred_train = []

        for X_train, y_train in train_data_loader:
            optimizer.zero_grad() # Zero gradients for every batch
            outputs = model(X_train) # Make predictions
            loss = loss_fn(outputs.view(-1),y_train)  # Calculate loss
            loss.backward() # Calculate gradients
            optimizer.step() # Update weights

            running_train_loss += loss.item()
            all_y_true_train.append(y_train.detach())
            all_y_pred_train.append(outputs.detach())

        
        epoch_train_mse = running_train_loss/len(train_data_loader)
        y_true_tensor = torch.cat(all_y_true_train)
        y_pred_tensor = torch.cat(all_y_pred_train)
        epoch_train_r2 = r2_score_torch(y_true_tensor, y_pred_tensor).item()

        running_test_loss = 0
        all_y_true_test = []
        all_y_pred_test = []

        with torch.no_grad():
            for X_test, y_test in test_data_loader:
                outputs = model(X_test) # Make predictions
                loss = loss_fn(outputs.view(-1),y_test) # Calculate loss

                running_test_loss += loss.item()
                all_y_true_test.append(y_train.detach())
                all_y_pred_test.append(outputs.detach())

        epoch_test_mse = running_test_loss/len(test_data_loader)
        y_true_tensor = torch.cat(all_y_true_test)
        y_pred_tensor = torch.cat(all_y_pred_test)
        epoch_test_r2 = r2_score_torch(y_true_tensor, y_pred_tensor).item()
        
        metrics["epoch"].append(epoch)
        metrics["train_mse"].append(epoch_train_mse)
        metrics["test_mse"].append(epoch_test_mse)
        metrics["train_r2"].append(epoch_train_r2)
        metrics["test_r2"].append(epoch_test_r2)

        print(f'Epoch {epoch},')
        print(f'Train: MSE = {epoch_train_mse}, R2 = {epoch_train_r2}')
        print(f'Test: MSE = {epoch_test_mse}, R2 = {epoch_test_r2}')
        print('')
        scheduler.step()

    with open(save_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        rows = zip(*metrics.values())
        writer.writerows(rows)
    print(f"Metrics saved to {save_path}")
            
                    