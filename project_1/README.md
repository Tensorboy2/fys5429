# Project 1, Convolutional Neural Network For Permeability Prediction
### By Sigurd Vargdal

## Structure
The code is separated into components.
- **cnn.py**: The Convolutional Neural Network (CNN) class. Built with Pytorch, has only a constructor for the architecture and the forward method for making predictions.
- **train.py**: The Trainer class. A class with a constructor containing the list of metrics being keep track of during training. It has one method **train()** which runs the training of the CNN.
- **plot.py**: The Plotter class. A class that takes the model and trainer classes and plots the various metrics and features for the model. Each plot is called with a class method and is shown by  ```python matplotlib.pyplot.show()```.
- **data_pipeline.py**: The data pipeline. A set of functions for generating the data set using lattice boltzmann simulations.
- **main.py**: Main program of **Project 1**. Controls and runs everything together.

How to run:
```bash
python3 main.py
```