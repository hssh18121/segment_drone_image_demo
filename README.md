
# Set up environment

## Install conda 
https://docs.anaconda.com/miniconda/

- Choose the installer that matches the OS
- Specify the miniconda path if needed


## Create new conda environment

- conda create -n myenv python=3.9
- conda init
- source activate base
- conda activate myenv

## Download necessary dependencies for the conda environment
- pip install opencv-python==4.9.0.80
- pip install numpy==1.26.4
- pip install matplotlib==3.8.4
- pip install patchify
- pip install segmentation_models==1.0.1
- pip install tensorflow==2.14.0

# How to run inference code

### Run via command line, provide necessary arguments followed this convention:
```
python inference.py {rgb_image_path} {saved_model_path} {segmented_output_directory}
```

### Example:
```
python inference.py ./rgb_image/image1.tif ./saved_models/upernet_model.tf ./segmented_output
```