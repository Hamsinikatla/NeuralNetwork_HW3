# NeuralNetwork_HW3

Q1.
# MNIST Autoencoder

This project demonstrates a simple autoencoder using TensorFlow and Keras to reconstruct MNIST handwritten digit images.

## Installation

pip install numpy matplotlib tensorflow

## Usage

Run the script to:
- Load and preprocess the MNIST dataset
- Train a basic autoencoder (784 → 32 → 784)
- Visualize original vs reconstructed images

## Code Overview

- **Data**: MNIST (flattened 28x28 images)
- **Model**: Fully connected autoencoder with one hidden (latent) layer
- **Loss**: Binary crossentropy
- **Optimizer**: Adam

##  Output

Displays a grid of original and reconstructed images using matplotlib.

##  Features & Learnings

- Dimensionality reduction with a latent space
- Unsupervised learning approach
- Visual comparison of input and output
