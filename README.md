# Handwritten Math Symbols Classification with CNN

This project implements a Convolutional Neural Network (CNN) to classify handwritten math symbols. The dataset is structured in folders, each representing a class, and a simple image classification task is performed using a CNN built with PyTorch.

## Overview
This project demonstrates a CNN-based approach to classify images of handwritten math symbols. It utilizes PyTorch for model implementation and training. The trained model can distinguish between 19 different symbols.

## Pretrained Model
A pretrained model with 90% validation accuracy is provided in the repository. You can directly run the Jupyter notebook without training the model again, as the model weights are already saved in the file handwritten_math_symbols_model.pth. I uploaded some trial data in the `trial_data` directory for testing purposes. You can add more data.

To train your own model, download all requirements, and the dataset, and run the `main.py` file.

## Dataset
The dataset is an image dataset structured with subfolders for each symbol class. Each folder contains images for that particular class, and the `ImageFolder` module from `torchvision` is used for loading the data.

- **Classes**: This project has 19 classes, including digits and mathematical operators such as "add," "sub," "mul," etc.
- **Data Splitting**: 80% of the data is used for training and 20% for validation.

To download the dataset, you can use the following code snippet:
```python
import kagglehub

path = kagglehub.dataset_download("sagyamthapa/handwritten-math-symbols")

print("Path to dataset files:", path)
```

Make sure to place the downloaded dataset files in a directory named 'dataset', as this is the path specified in the code.

## Model Architecture
The CNN model architecture used here is as follows:
1. **Convolutional Layers**: 
   - 4 convolutional layers with increasing depth: [3 → 32, 32 → 64, 64 → 64, 64 → 128]
   - `ReLU` activation function applied after each convolution.
   - Max pooling after the second and third convolutions.
2. **Fully Connected Layers**:
   - 1 fully connected layer of 64 neurons with `ReLU` and dropout for regularization.
   - Output layer with 19 neurons (one for each class).
3. **Pooling**: Max pooling layers with kernel size 2.
4. **Dropout**: Dropout with a rate of 0.5 applied after the first fully connected layer.

This architecture is designed for image input of size 32x32 and performs well on this classification task.

## Installation and Requirements
To run this project, install the following dependencies:
- `torch`: `pip install torch`
- `torchvision`: `pip install torchvision`
- `numpy`: `pip install numpy`
- `matplotlib` (for visualization): `pip install matplotlib`
