
# Skibidi Network: A From-Scratch Neural Network for MNIST Digit Recognition 🤖✨

This repository presents an implementation of a simple neural network built entirely from scratch using Python and NumPy. The network is designed to perform classification on the MNIST dataset, which consists of 28×28 pixel grayscale images of handwritten digits.

## Description

The neural network in this repository features a straightforward architecture with the following components:

- **Input Layer:**  
  Accepts 784 features corresponding to the flattened 28×28 pixel images.

- **Hidden Layer:**  
  Consists of 64 neurons. Each neuron applies the hyperbolic tangent (tanh) activation function. Weights are initialized randomly from the interval \([-0.5, 0.5]\) and each neuron receives a bias of 0.5.

- **Output Layer:**  
  Comprises 10 neurons, one for each digit from 0 to 9. The tanh activation function is applied at this layer, with a bias of 0.7 assigned to each neuron.

> **Note:** Although using the tanh activation function and Mean Squared Error (MSE) loss is not the best approach for digit classification (a softmax activation with cross-entropy loss is generally preferred), we are using them here solely for educational purposes. 🎓

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

## Libraries Used

- **TensorFlow:**  
  Utilized exclusively for loading the MNIST dataset.

- **NumPy:**  
  Employed for matrix operations and numerical computations throughout the network.

- **Matplotlib:**  
  Used for creating visualizations of the training loss and sample predictions.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- TensorFlow (only for loading the MNIST dataset)

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/a1harfoush/NN-from-scratch.git
cd NN-from-scratch
```

Install the required dependencies:

```bash
pip install numpy matplotlib tensorflow
```

## Usage

Run the main script to train the neural network and visualize the results:

```bash
python nn_mnist.py
```

Upon execution, the script will:
- Train the network on the MNIST dataset.
- Output the training progress and final test accuracy.
- Plot the training loss over epochs 📈.
- Display a selection of test images with their predicted and true labels.

## Results & Visualization

After training, you will see:

- **Test Accuracy:**  
  The script will output the overall test accuracy (e.g., `Test accuracy: 0.76`). This provides a quantitative measure of the network's performance.

- **Visualization:**

  - A **Training Loss Plot** that shows how the loss decreases over epochs.
  
     ![download](https://github.com/user-attachments/assets/af60cf50-36e8-4122-b79d-aa10bf11e059)

  - **Sample Predictions:** A visual gallery of selected test images alongside their true and predicted labels.
 
     ![download](https://github.com/user-attachments/assets/ae3ff6d0-d5ed-4249-8ebb-2af21573a0cc)


Feel free to update this section with your own results and screenshots once you've run the network!

## Acknowledgements

- The MNIST dataset is provided by Yann LeCun and collaborators.
- This project is intended for educational purposes, illustrating the fundamental concepts of neural network implementation and training.
