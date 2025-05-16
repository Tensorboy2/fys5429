'''Training module for CNN.'''
import torch
import torch.optim as op
import pandas
import os
path = os.path.dirname(__file__)
import math

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
          warmup_steps=0,
          decay="",
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
    num_batches = len(train_data_loader)
    total_steps = num_epochs*(len(train_data_loader))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1)/ warmup_steps
        else:
            decay_epochs = total_steps - warmup_steps
            decay_progress = (step - warmup_steps) / decay_epochs
            if decay=="linear":
                return max(0.0, 1.0 - decay_progress)
            elif decay=="cosine":
                return 0.5 * (1 + math.cos(math.pi * decay_progress))
            else:
                return 1

    scheduler = op.lr_scheduler.LambdaLR(optimizer,lr_lambda)



    for epoch in range(num_epochs):
        '''Training:'''
        model.train()
        running_train_loss = 0
        all_y_true_train = []
        all_y_pred_train = []

        # print(f"[{model_name}] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(f"[{model_name}] Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        for batch_idx, (image_train, image_filled_train, k_train) in enumerate(train_data_loader):
            image_train, image_filled_train, k_train = image_train.to(device), image_filled_train.to(device), k_train.to(device) # Send to device

            optimizer.zero_grad() # Zero gradients for every batch
            outputs_image = model(image_train) # Make predictions
            outputs_image_filled = model(image_filled_train) # Make predictions

            loss =  (loss_fn(outputs_image,k_train)+ loss_fn(outputs_image_filled,k_train))/2 # Calculate loss

            loss.backward() # Calculate gradients
            optimizer.step() # Update weights

            running_train_loss += loss.item()
            all_y_true_train.append(k_train.detach())
            all_y_pred_train.append(((outputs_image+outputs_image_filled)/2).detach())

            scheduler.step()
            print(f"Batch {batch_idx}/{num_batches}",flush=True)

        
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
            for image_test, image_filled_test, k_test in test_data_loader:
                image_test, image_filled_test, k_test = image_test.to(device), image_filled_test.to(device), k_test.to(device)
                outputs_image = model(image_test) # Make predictions
                outputs_image_filled = model(image_filled_test) # Make predictions

                loss =  (loss_fn(outputs_image,k_test)+ loss_fn(outputs_image_filled,k_test))/2 # Calculate loss

                running_test_loss += loss.item()
                all_y_true_test.append(k_test.detach())
                all_y_pred_test.append(((outputs_image+outputs_image_filled)/2).detach())

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
        torch.cuda.empty_cache()

    df = pandas.DataFrame(metrics)
    df.to_csv(os.path.join(path,"results",save_path))
    print(f"Metrics saved to {save_path}")
            
                    