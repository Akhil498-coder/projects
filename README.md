# projects
CIFAR-10 CNN Image Classification
This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. This project includes functionality to train the model on subsets of the dataset.

Features
Custom CNN architecture.

Training on partial CIFAR-10 datasets.

Balanced data subset creation.

GPU acceleration (if available).

Training progress visualization.

Model checkpointing.

CIFAR-10 Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship, and truck.

Getting Started
Prerequisites
Python 3.x

PyTorch

Torchvision

Matplotlib

NumPy

Install dependencies:

pip install torch torchvision matplotlib numpy

Running the Code
Clone this repository.

Open the project_image.ipynb (or your Python script).

Adjust configuration parameters like DATASET_FRACTION or TRAIN_SAMPLES as needed.

Run the script.

The script will download the dataset, train the model, and display results.

Project Structure
.
├── project_image.ipynb
└── cifar10_models/
    ├── best_model_partial.pth
    └── checkpoint_partial_epoch_X.pth

Results
Training output will show dataset usage, total time, and final accuracy.

Example:

TRAINING COMPLETE!
Dataset used: 10,000 train, 2,000 test samples
Percentage of full dataset: 20.0%
Total time: 1.4 minutes
Best test accuracy: 59.00%
Final test accuracy: 58.10%

License
This project is open-source under the MIT License.
