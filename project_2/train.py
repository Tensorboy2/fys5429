'''Training module for CNN.'''
import torch
import torch.optim as op
import pandas
import os
path = os.path.dirname(__file__)

@torch.jit.script
def r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Jit'ed rd score function
    """
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    
    mean_y_true = torch.mean(y_true)
    ss_tot = torch.sum((y_true - mean_y_true) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    
    # Prevent divide by zero
    if ss_tot < 1e-8:
        return torch.tensor(0.0, device=y_true.device)
    
    r2 = 1 - ss_res / ss_tot
    return r2



def train(model = None, 
          optimizer = None, 
          train_data_loader = None, 
          test_data_loader = None, 
          num_epochs = 5, 
          device = 'cpu',
          save_path="metrics.csv",
          warmup_epochs=0,
          save_model_path=None):
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

    if save_model_path is None:
        model_name = getattr(model, "name", model.__class__.__name__)
        save_model_path = f"{model_name}.pth"

    best_test_mse = float("inf")
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1)/ warmup_epochs
        else:
            return 1

    scheduler = op.lr_scheduler.LambdaLR(optimizer,lr_lambda)

    for epoch in range(num_epochs):
        '''Training:'''
        model.train()
        running_train_loss = 0
        all_y_true_train = []
        all_y_pred_train = []

        for X_train, y_train in train_data_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)

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

        '''Evaluation:'''
        model.eval()
        running_test_loss = 0
        all_y_true_test = []
        all_y_pred_test = []

        with torch.no_grad():
            for X_test, y_test in test_data_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                outputs = model(X_test) # Make predictions
                loss = loss_fn(outputs.view(-1),y_test) # Calculate loss

                running_test_loss += loss.item()
                all_y_true_test.append(y_test.detach())
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

        if (epoch_test_mse<best_test_mse):
            '''Save best performing model:'''
            best_test_mse = epoch_test_mse
            torch.save(model.state_dict(), os.path.join(path,"models",save_model_path))
            print(f"Best model form epoch: {epoch} with MSE: {best_test_mse:.5f} saved.")

        print(f'Epoch {epoch},')
        print(f'Train: MSE = {epoch_train_mse}, R2 = {epoch_train_r2}')
        print(f'Test: MSE = {epoch_test_mse}, R2 = {epoch_test_r2}')
        print('')

        scheduler.step()

    df = pandas.DataFrame(metrics)
    df.to_csv(os.path.join(path,"results",save_path))
    print(f"Metrics saved to {save_path}")
            
                    