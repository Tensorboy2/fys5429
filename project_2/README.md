# Project 2
## By Sigurd Vargdal

Things to include:
- Data generation
    - Use of make media file
    - Use of *data_pipeline_2.py* 
    - Convert tp *.npz* format
- Modules in models
    - ResNets
    - ViTs
    - ConvNeXts?
- Training procedure
    - Warmup
    - Decay (LR and Weight)
    - Data augmentation
- Config usage
    - Basic setup

For running training:
```yaml
experiments:
  - model: Model_Name
    save_path: "path_to_save_results.csv"
    hyperparameters:
      num_epochs: 30 # Beyond paper just to check
      lr: 0.0008 # Based on paper
      warmup_steps: 300 # Based on batch size times num data samples
      weight_decay: 0.1 # Based on paper
      batch_size: 128 # What the GPU could handel and stable training
      decay: "linear" # Options are ("","linear","cosine")
    data:
      hfli: True # Horizontal flip data augmentation
      vfli: True # Vertical flip data augmentation
      rotate: True # Rotation data augmentation
      num_samples: null # null means all data
      test_size: 0.2 # Default
```


If data exists the *main.py* can be run with
```bash
python3 main.py configs/example_config.yaml
```
(Optional add `bash > example_log.txt` for logging print statements)