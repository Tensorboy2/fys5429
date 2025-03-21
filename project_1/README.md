# Project 1, Convolutional Neural Network For Permeability Prediction
### By Sigurd Vargdal

This project begins with a data pipe line which can produce porous media images and using Lattice Boltzmann simulations the porous media permeability. The code use MPI to perform analysis in parallel. To run us one of these examples:
```bash
mpirun -np 8 python3 project_1/data_pipeline.py
```
If 8 cores are available. (For -np 8 on 16 cores. 1,000 samples takes about 1 hour. The code is set to generate 10,000 data points)

After this, one can run the main function. This function includes 4 main parts. Two function for performing grid searches on the CNN and the FFNN, and then two functions that can run longer training on the best set of hyper parameters (These parameters must be implemented manually). To run: 
```bash
python3 project_1/main.py
```

Once the data is processed the data from the training can be visualized using the plot script. This produces plots of both the grid searches and the two longer searches. Additionally, the code gives a plot of the prediction of the long trained models. To run:
```bash
python3 project_1/plot.py
```
