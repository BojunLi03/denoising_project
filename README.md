# denoising
This is a denoising pipeline, designed to evaluate the performance of various loss functions on medical image denoising. The pipeline contains `loss_functions.py` (the file that contains the implementations of various different loss functions), `denoising_pipeline.ipynb` (the file that runs the training and tests), and `plot_curves.ipynb` (a helper file that visualizes loss curves on already-trained models). A `loss_function_notes.csv` is also provided for reference.

## Installation

First, clone this repository onto your local machine: `git clone https://github.com/BojunLi03/denoising_project.git`.

### Install dependencies

This pipeline relies on a number of imported Python libraries to function. To install, go into the environment you prefer to work in, and run `pip install -r requirements.txt`.

## Using the pipeline

To use the pipeline, you should have a cuda-enabled device, otherwise the notebook will not work.

The pipeline doesn't come with data; you must provide it with a data file (the code assumes the file is named `data_file_name.h5` and is located in the same directory as the .ipynb and .py files). Once you have a data file (and have modified the `data_file_name` variable to be your data file's relative path), you may execute the notebook.

### User-defined Parameters

The pipeline, at the top of `denoising_pipeline.ipynb`, provides a number of user-defined parameters, including a learning rate parameter, as well as parameters for where to save trained models.
