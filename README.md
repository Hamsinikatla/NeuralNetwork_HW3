# NeuralNetwork_HW3

# Q1.Implementing a Basic Autoencoder
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





# Q2.Implementing a Denoising Autoencoder
# Denoising Autoencoder for MNIST

This project demonstrates how to build a **denoising autoencoder** using **TensorFlow** and **Keras** to remove noise from MNIST handwritten digit images.

## Installation

```bash
pip install numpy matplotlib tensorflow
```

## Usage

Run the script to:
- Load and preprocess the MNIST dataset
- Add Gaussian noise to the images
- Train a denoising autoencoder (784 → 32 → 784)
- Visualize noisy, clean, and denoised images

## Code Overview

- **Data**: MNIST (flattened 28x28 images)
- **Noise**: Gaussian noise (mean=0, std=0.5)
- **Model**: Fully connected denoising autoencoder with one hidden (latent) layer
- **Loss**: Binary crossentropy
- **Optimizer**: Adam

## Output

Displays a grid with:
- **Top row**: Noisy images.
- **Middle row**: Clean ground truth images.
- **Bottom row**: Denoised images from the model.

## Features & Learnings

- Denoising images by training the autoencoder on noisy data.
- Unsupervised learning with an emphasis on image noise removal.
- Visual comparison of noisy, clean, and denoised images.





# Q3 Implementing an RNN for Text Generation

Overview
This project demonstrates the implementation of a Recurrent Neural Network (RNN) using Long Short-Term Memory (LSTM) for generating text based on an input seed. The model is trained on a sample text from Shakespeare's Sonnet 18 ("Shall I compare thee to a summer's day?"). The text generation process is influenced by the concept of "temperature," which controls the randomness of the predictions, enabling varied results with the same seed text.

Installation Instructions
To run this project, you need Python 3.x and the following libraries installed:

TensorFlow (for deep learning functionality)

NumPy (for numerical operations)

You can install these libraries using pip:

pip install tensorflow numpy

Usage Guide
Load the Script: Copy the provided code into a Python file (e.g., text_generation_rnn.py).

Run the Script: After ensuring you have the necessary libraries installed, run the Python script.

Text Generation: The script will train a simple LSTM-based model on the input text and generate new text starting from a given seed.

Example command:

python text_generation_rnn.py

Code Explanation
Text Preprocessing:

The text is tokenized, meaning each unique character is assigned a unique index. A dictionary is created to map each character to an index and vice versa.

Input sequences of length seq_length are created to serve as training data for the LSTM model, where each sequence has an associated next character (label).

Model Architecture:

An LSTM model is defined using Keras layers:

Embedding: Converts character indices to dense vectors of fixed size (8).

LSTM: A recurrent layer with 128 units that processes the sequences.

Dense: A fully connected output layer with softmax activation to predict the next character in the sequence.

The model is compiled using categorical cross-entropy as the loss function and Adam as the optimizer.

Training:

The model is trained for 50 epochs using the preprocessed sequences and labels.

Text Generation:

A generate_text function generates new text from a seed string. It uses a temperature parameter to control randomness:

Lower temperatures result in more predictable text.

Higher temperatures lead to more diverse text.

Output:

The script generates text starting from the seed phrase "Shall I com" and prints it to the console.






# Q4 Sentiment Classification Using RNN

Overview
This project implements sentiment classification on the IMDB movie reviews dataset using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) units. The model predicts whether a given review is positive or negative based on its text. The model uses embeddings to represent words as vectors and LSTM layers to capture sequential patterns in the text data.

Installation Instructions
To run this project, ensure that you have Python 3.x installed and the following libraries:

TensorFlow (for deep learning functionality)

Scikit-learn (for metrics evaluation)

NumPy (for numerical operations)

You can install the required libraries using pip:

pip install tensorflow scikit-learn numpy

Usage Guide
Load the Script: Copy the provided code into a Python file (e.g., sentiment_classification_rnn.py).

Run the Script: Ensure you have the necessary libraries installed and run the script.

Train and Evaluate: The model will be trained on the IMDB dataset, and the confusion matrix and classification report will be displayed.

Example command:

python sentiment_classification_rnn.py
Code Explanation
Load IMDB Dataset:

The IMDB dataset is loaded using TensorFlow's imdb.load_data() method. The dataset is pre-processed to retain the top 10,000 most frequent words.

The dataset is split into training and test sets, where the reviews are represented as sequences of integers corresponding to words.

Data Preprocessing:

The sequences of integers (reviews) are padded to a uniform length of 250 words using pad_sequences. This ensures that each input sequence to the model is of the same length.

Model Architecture:

The model is built using a Sequential approach:

Embedding Layer: Converts word indices into dense vectors of fixed size (128 in this case).

LSTM Layer: A Long Short-Term Memory (LSTM) layer with 128 units, which captures the sequential nature of the text. Dropout and recurrent dropout are applied to reduce overfitting.

Dense Layer: A final dense layer with a sigmoid activation function, suitable for binary classification (positive/negative sentiment).

Model Compilation:

The model is compiled using the binary cross-entropy loss function, which is appropriate for binary classification problems. The Adam optimizer is used for training.

Model Training:

The model is trained on the training data for 3 epochs. You can adjust the number of epochs for longer training.

The validation split is set to 0.2 to reserve 20% of the training data for validation during training.

Model Evaluation:

After training, the model's performance is evaluated on the test set. The probabilities predicted by the model are converted into binary labels (positive or negative) using a threshold of 0.5.

A confusion matrix and classification report (including accuracy, precision, recall, and F1-score) are generated to evaluate the model's performance.

Confusion Matrix and Classification Report:

The confusion matrix shows the number of true positives, true negatives, false positives, and false negatives.

The classification report provides detailed metrics for evaluating the model's performance.
output shows:

The confusion matrix, which shows the performance of the classifier in terms of the true positive, false positive, true negative, and false negative counts.

The classification report, which includes:

Precision: The ratio of correct positive predictions to the total predicted positives.

Recall: The ratio of correct positive predictions to the total actual positives.

F1-Score: The harmonic mean of precision and recall.

Accuracy: The overall proportion of correct predictions.

Summary of Outputs
Confusion Matrix: A table that displays the counts of true positives, false positives, true negatives, and false negatives.

Classification Report: Includes detailed performance metrics such as precision, recall, F1-score, and accuracy.

The model’s performance is assessed based on these metrics, with a balance between positive and negative sentiment predictions.

Key Learnings
LSTM for Sentiment Analysis: LSTM networks are effective for text classification tasks because they capture the sequential nature of language.

Evaluation Metrics: Understanding metrics like accuracy, precision, recall, and F1-score is crucial for evaluating the performance of a binary classification model.

Data Preprocessing: Padding and truncating sequences to a fixed length ensures uniformity in the input data, which is essential for feeding into neural networks.

Features
Binary Sentiment Classification: The model classifies movie reviews as either positive or negative.

LSTM-based Architecture: Utilizes LSTM, a type of RNN, to handle the sequential dependencies in text data.

Dropout for Regularization: The model uses dropout to prevent overfitting.

Evaluation Tools: The script generates a confusion matrix and a classification report to assess model performance.
