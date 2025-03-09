'''Training module for CNN.'''
import torch
import torch.optim as op
from sklearn.metrics import r2_score, mean_absolute_error

class Trainer():
    '''
    Class for training a cnn.
    '''
    def __init__(self):
        '''
        The constructor initializes different metrics that can be extracted later.
        Doing this in Python introduces a lot of over head during the computation, but 
        it can be more convenient.  
        '''

        # Initializing training metrics.
        self.train_mse = [] # Mean square error
        # self.train_mae = [] # Mean absolute error
        self.train_r2 = [] # Coefficient of Determination

        # Initializing training metrics.
        self.test_mse = [] # Mean square error
        # self.test_mae = [] # Mean absolute error
        self.test_r2 = [] # Coefficient of Determination

    
    def train(self, model = None, optimizer = None, train_data_loader = None, test_data_loader = None, num_epochs = 5):
        '''
        Training method for training.

        '''
        loss_fn = torch.nn.MSELoss()
        for epoch in range(num_epochs):
            running_train_loss = 0
            # running_train_mae = 0
            running_train_r2 = 0
            for X_train, y_train in train_data_loader:
                # X_train, y_train = train_data

                optimizer.zero_grad() # Zero gradients for every batch

                outputs = model(X_train) # Make predictions

                loss = loss_fn(outputs.view(-1),y_train) # Calculate loss
                loss.backward() # Calculate gradients

                optimizer.step() # Update weights

                running_train_loss += loss.item()
                y_pred = outputs.detach().numpy()
                # running_train_mae += mean_absolute_error(y_true=y_train,y_pred=y_pred)
                running_train_r2 += r2_score(y_true=y_train,y_pred=y_pred)
            
            epoch_train_mse = running_train_loss/len(y_train)
            self.train_mse.append(epoch_train_mse)

            # epoch_train_mae = running_train_loss/len(y_train)
            # self.train_mae.append(epoch_train_mae)

            epoch_train_r2 = running_train_r2/len(y_train)
            self.train_r2.append(epoch_train_r2)
 
            running_test_loss = 0
            # running_train_mae = 0
            running_test_r2 = 0
            with torch.no_grad():
                for X_test, y_test in test_data_loader:
                    outputs = model(X_test) # Make predictions

                    loss = loss_fn(outputs.view(-1),y_test) # Calculate loss
                    running_test_loss += loss.item()
                    y_pred = outputs.detach().numpy()
                    # running_train_mae += mean_absolute_error(y_true=y_train,y_pred=y_pred)
                    running_test_r2 += r2_score(y_true=y_test,y_pred=y_pred)

            epoch_test_mse = running_test_loss/len(y_test)
            self.test_mse.append(epoch_test_mse)

            # epoch_train_mae = running_train_loss/len(y_train)
            # self.train_mae.append(epoch_train_mae)

            epoch_test_r2 = running_test_r2/len(y_test)
            self.test_r2.append(epoch_test_r2)
            
            print(f'Epoch {epoch},')
            print(f'Train: MSE = {epoch_train_mse}, R2 = {epoch_train_r2},')
            print(f'Test: MSE = {epoch_test_mse}, R2 = {epoch_test_r2}')
            print('')

def train(model = None, optimizer = None, train_data_loader = None, test_data_loader = None, num_epochs = 5, l1 = 0, l2 = 0):
    '''
    Training function.

    '''
    train_mse = [] # Mean square error
    train_mae = [] # Mean absolute error
    train_r2 = [] # Coefficient of Determination
    test_mse = [] # Mean square error
    test_mae = [] # Mean absolute error
    test_r2 = [] # Coefficient of Determination
    loss_fn = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        running_train_loss = 0
        running_train_mae = 0
        running_train_r2 = 0
        for X_train, y_train in train_data_loader:
            # X_train, y_train = train_data

            optimizer.zero_grad() # Zero gradients for every batch

            outputs = model(X_train) # Make predictions
            loss = loss_fn(outputs.view(-1),y_train)  # Calculate loss

            if l1!=0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1*l1_norm # Add L2 penalty
            if l2!=0:
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss += l2*l2_norm # Add L2 penalty
            loss.backward() # Calculate gradients

            optimizer.step() # Update weights

            running_train_loss += loss.item()
            y_pred = outputs.detach().numpy()
            running_train_mae += mean_absolute_error(y_true=y_train,y_pred=y_pred)
            running_train_r2 += r2_score(y_true=y_train,y_pred=y_pred)
        
        epoch_train_mse = running_train_loss/len(y_train)
        train_mse.append(epoch_train_mse)

        epoch_train_mae = running_train_mae/len(y_train)
        train_mae.append(epoch_train_mae)

        epoch_train_r2 = running_train_r2/len(y_train)
        train_r2.append(epoch_train_r2)

        running_test_loss = 0
        running_test_mae = 0
        running_test_r2 = 0
        with torch.no_grad():
            for X_test, y_test in test_data_loader:
                outputs = model(X_test) # Make predictions

                loss = loss_fn(outputs.view(-1),y_test) # Calculate loss
                running_test_loss += loss.item()
                y_pred = outputs.detach().numpy()
                running_test_mae += mean_absolute_error(y_true=y_test,y_pred=y_pred)
                running_test_r2 += r2_score(y_true=y_test,y_pred=y_pred)

        epoch_test_mse = running_test_loss/len(y_test)
        test_mse.append(epoch_test_mse)

        epoch_test_mae = running_test_mae/len(y_test)
        test_mae.append(epoch_test_mae)

        epoch_test_r2 = running_test_r2/len(y_test)
        test_r2.append(epoch_test_r2)
        
        print(f'Epoch {epoch},')
        print(f'Train: MSE = {epoch_train_mse}, R2 = {epoch_train_r2}, MAE = {epoch_train_mae}')
        print(f'Test: MSE = {epoch_test_mse}, R2 = {epoch_test_r2}, MAE = {epoch_test_mae}')
        print('')
    return train_mse, test_mse, train_r2, test_r2, train_mae, test_mae
            
                    


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    mse = trainer.train_mse