# Project 1, Convolutional Neural Network For Permeability Prediction
### By Sigurd Vargdal

## Structure
The code is separated into components.
- **cnn.py**: The Convolutional Neural Network (CNN) class. Built with Pytorch, has only a constructor for the architecture and the forward method for making predictions.
- **train.py**: The Trainer class. A class with a constructor containing the list of metrics being keep track of during training. It has one method **train()** which runs the training of the CNN.
- **main.py**: Main program of **Project 1**. Controls and runs everything together.

How to run:
```bash
python3 main.py
```