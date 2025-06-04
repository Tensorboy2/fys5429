# Data
Inside this folder is the code for the data generation and some codes to produce demo plots fo various parts of the data and process. 

## **data_generation/**:
For the generation of the porous media domains, **make_images.py** is used. It uses **percolation.py** to check percolation condition for each domain. It uses a slightly stricter version of the Newmann-Ziff algorithm, requiering full in-domain connected paths. This introduces some false-negatives, which are demoed in the **plot_percolation_results** function with the **generate_cross_channels** domain.
*see [https://github.com/Tensorboy2/fys5429/blob/main/project_2/README.md](https://github.com/Tensorboy2/fys5429/blob/main/project_2/README.md) for execution details.*