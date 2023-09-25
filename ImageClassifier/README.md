# Image Classification with Python Scripts

This repository contains two Python scripts, `train.py` and `predict.py`, for training and using an image classification model respectively. These scripts are designed to be run from the terminal with various command-line options. Below, you'll find instructions on how to use these scripts along with some examples.

## Prerequisites

Before using these scripts, ensure you have the following prerequisites installed:
- Python (3.6 or higher)
- PyTorch
- torchvision
- CUDA
- Matplotlib
- Seaborn
- NumPy

## Training a Model (train.py)

To train a model, use the `train.py` script. Here are the available command-line arguments:

- `data_dir` (mandatory): Provide the directory containing the training data.
- `--save_dir` (optional): Specify the directory to save the trained model.
- `--arch` (optional): Specify the neural network architecture (default is densenet121, but you can also use alexnet).
- `--lrn` (optional): Set the initial learning rate (default is 0.002).
- `--hidden_units` (optional): Number of hidden units in the classifier (default is 2048).
- `--epochs` (mandatory): Maximum number of training epochs. Training stops when validation accuracy reaches 90% or the specified number of epochs.
- `--GPU` (optional): Specify to use GPU for training.

**Example:**

```bash
python train.py 'flowers' --save_dir 'checkpoint' --arch densenet121 --lrn 0.001 --hidden_units 512 --epochs 20 --GPU
```

This command trains a model on the data in the "flowers" directory using the densenet121 architecture, with a learning rate of 0.001, 512 hidden units, for a maximum of 20 epochs, and utilizes the GPU.

## Making Predictions (predict.py)
To make predictions on an image, use the predict.py script. Here are the available command-line arguments:

- `image_dir` (mandatory): Provide the path to the image you want to predict.
- `load_dir` (mandatory): Provide the path to the checkpoint file of the trained model.
- `--top_k` (optional): Number of top K most likely classes to display.
- `--category_names` (optional): JSON file containing the mapping of categories to real names.
- `--GPU` (optional): Specify to use GPU for prediction.

**Example:**

```bash
python predict.py 'flowers/test/1/image_06743.jpg' 'checkpoints/model.pth' --top_k 3 --category_names 'cat_to_name.json' --GPU
```

This command predicts the top 3 most likely classes for the image "image_06743.jpg" using the trained model saved in "model.pth" and displays the results with category names from the "cat_to_name.json" file while utilizing the GPU.











