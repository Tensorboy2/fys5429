import pandas as pd
import matplotlib.pyplot as plt
import re

# Prepare storage
epochs = []
train_mse = []
train_r2 = []
test_mse = []
test_r2 = []

# Read and parse the file
with open("train_data.txt", "r") as f:
    lines = f.readlines()

for i in range(0, len(lines), 3):  # every 3 lines = 1 epoch block
    try:
        epoch_line = lines[i].strip()
        train_line = lines[i+1].strip()
        test_line = lines[i+2].strip()

        # Extract epoch number
        epoch = int(re.search(r'Epoch (\d+)', epoch_line).group(1))

        # Extract train MSE and R2
        train_mse_val = float(re.search(r'MSE = ([\d.]+)', train_line).group(1))
        train_r2_val = float(re.search(r'R2 = ([\d.-]+)', train_line).group(1))

        # Extract test MSE and R2
        test_mse_val = float(re.search(r'MSE = ([\d.]+)', test_line).group(1))
        test_r2_val = float(re.search(r'R2 = ([\d.-]+)', test_line).group(1))

        # Store values
        epochs.append(epoch)
        train_mse.append(train_mse_val)
        train_r2.append(train_r2_val)
        test_mse.append(test_mse_val)
        test_r2.append(test_r2_val)

    except Exception as e:
        print(f"Skipping block at line {i}: {e}")

# Make DataFrame
df = pd.DataFrame({
    "epoch": epochs,
    "train_mse": train_mse,
    "train_r2": train_r2,
    "test_mse": test_mse,
    "test_r2": test_r2
})

print(df.head())

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_mse'], label='Train MSE')
plt.plot(df['epoch'], df['test_mse'], label='Test MSE')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training and Test MSE over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_r2'], label='Train R2')
plt.plot(df['epoch'], df['test_r2'], label='Test R2')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training and Test R2 over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
