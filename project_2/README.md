# Project 2
## By Sigurd Vargdal
This project aims to compare deep learning architectures on predicting permeability from a 2D porous medium. Architectures include ResNet, ViT and ConvNeXt.

This project contains three main parts:
- Data
- Training
- Plotting

## Data
The data folder of this project contains the full process of generating the required dataset for training the models. 
The first code: **make_images.py** uses the modified binary\_blobs function with a percolation check to generate synthetic 2D periodic porous media. 
```bash
python3 data/data_generation/make_images.py
```

The next step is: **data_pipeline_2.py**, which uses **simulation/lbm.py** to execute the Lattice-Boltzmann simulations on the generated domains.
```bash
mpirun -np 8 data/data_pipeline_2.py
```
The data generated can be converted manually to **.npz** format for faster transfer to remotes or clusters.
*images.npz*, *images\_filled.npz* and *k.npz* are required for the data loader so this must be made manually.
Configure the folder that needs to be read to npz and then run:
```bash
python3 data/convert_to_npz.py
```

## Training
To reproduce the training done in the project there are 3 runs within **configs/**:
- **dataaugmentation_test.yaml** 
- **diff_num_datapoints.yaml** 
- **all_models.yaml** 

Each config file has the following general structure:
```yaml
experiments:
  - model: Model_Name
    save_path: "path_to_save_results.csv"
    hyperparameters:
      num_epochs: 30 # Beyond paper just to check
      lr: 0.0008 # Based on paper
      warmup_steps: 1000 # Based on batch size times num data samples
      weight_decay: 0.1 # Based on paper
      batch_size: 128 # What the GPU could handel and stable training
      decay: "cosine" # Options are ("","linear","cosine")
    data:
      hflip: True # Horizontal flip data augmentation
      vflip: True # Vertical flip data augmentation
      rotate: True # Rotation data augmentation
      group: False # Group based agumentation (If True, overrides other augmentations)
      num_samples: null # null means all data
      test_size: 0.2 # Default
```


If data exists the *main.py* can be run with
```bash
python3 main.py "configs/example_config.yaml"
```
(Optional add `bash > example_log.txt` for logging print statements)

This produces the resulting **.csv** files in the **results/** folder. 
The project was executed on a remote cluster an ssh scp was used to transfer the data, and put in the right folder within results. This needs to be done manually.

## Plotting
There are 4 plotting functions for the results of this projects:
- Training results:
  - **dataaug/plot_dataaug.py**
  - **diff_num_datapoints/plot_diff_num_datapoints.py**
  - **full_training/plot_full_training.py**
- Predictions:
  - **plot_predictions.py**