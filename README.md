
# Skibidi Network: A From-Scratch Neural Network for MNIST Digit Recognition

This repository presents an implementation of a simple neural network built entirely from scratch using Python and NumPy. The network is designed to perform classification on the MNIST dataset, which consists of 28×28 pixel grayscale images of handwritten digits.

## Description

The neural network in this repository features a straightforward architecture with the following components:

- **Input Layer:**  
  Accepts 784 features corresponding to the flattened 28×28 pixel images.

- **Hidden Layer:**  
  Consists of 64 neurons. Each neuron applies the hyperbolic tangent (tanh) activation function. Weights are initialized randomly from the interval \([-0.5, 0.5]\) and each neuron receives a bias of 0.5.

- **Output Layer:**  
  Comprises 10 neurons, one for each digit from 0 to 9. The tanh activation function is also applied at this layer, with a bias of 0.7 assigned to each neuron.

The network is trained using Mean Squared Error (MSE) as the loss function, and the weights and biases are updated using gradient descent optimization. Additionally, the repository provides functionality to visualize training loss over epochs and to display sample predictions alongside their true labels.

## Features

- **From-Scratch Implementation:**  
  Demonstrates the fundamental principles of neural network design without relying on high-level machine learning libraries.

- **Custom Weight Initialization:**  
  Weights are randomly initialized within a specified interval, and biases are set to predetermined values.

- **Forward and Backward Propagation:**  
  Implements the forward pass with tanh activation and backpropagation using the derivative of tanh.

- **MNIST Digit Recognition:**  
  Trains the network on the MNIST dataset for the purpose of digit classification.

- **Visualization Tools:**  
  Includes code to plot the training loss over epochs and to visualize sample predictions on test images.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- TensorFlow (only for loading the MNIST dataset)

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/skibidi-network.git
cd skibidi-network
```

Install the required dependencies:

```bash
pip install numpy matplotlib tensorflow
```

## Usage

Run the main script to train the neural network and visualize the results:

```bash
python skibidi_network.py
```

Upon execution, the script will train the network on the MNIST dataset, output the training progress, display a plot of the training loss over epochs, and show a selection of test images along with their predicted and true labels.

## Acknowledgements

- The MNIST dataset is provided by Yann LeCun and collaborators.
- This project is intended for educational purposes, illustrating the fundamental concepts of neural network implementation and training.
