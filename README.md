# Skibidi Network: From Scratch to MNIST Awesomeness

Hey there, gorgeous coder! Welcome to the Skibidi Network repo—a playful, from-scratch neural network that learns to recognize those adorable MNIST digits. This project is all about mixing a little Python magic, NumPy wizardry, and our signature tanh flair to create something truly lit. Whether you're a coding newbie or a seasoned AI aficionado, get ready to ride the neural wave with style!

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Visualization](#results--visualization)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Ever dreamed of building your own neural network from the ground up? Our Skibidi Network is here to make that dream come true—no black-box libraries, just raw Python and NumPy vibes. This little beast features:
- **Input Layer:** Flattened MNIST images (28×28 pixels → 784 features).
- **Hidden Layer:** 64 neurons with a splash of tanh activation.
- **Output Layer:** 10 neurons (one for each digit 0–9) with our signature tanh magic.
- **Custom Initialization:** Weights set randomly in the range \([-0.5, 0.5]\), and biases with a cozy \(b1=0.5\) and \(b2=0.7\).

## Features

- **DIY Neural Network:** Build your own brainy machine with minimal dependencies.
- **MNIST Training:** Learn the art of digit recognition with one of the most classic datasets.
- **Visual Feedback:** Check out training loss curves and sample predictions to see the magic in action.
- **Clean & Playful Code:** Well-commented and structured with a twist of fun, perfect for learning and tweaking.

## Project Overview

This project guides you through every step of the network’s journey:
1. **Data Preprocessing:** Load, flatten, and normalize the MNIST images.
2. **Network Initialization:** Randomly set weights and biases to kick things off.
3. **Forward Pass:** Compute activations using tanh to introduce non-linearity.
4. **Loss Calculation:** Use Mean Squared Error (MSE) to measure our network’s "mood."
5. **Backpropagation:** Apply gradient descent to update weights and biases.
6. **Evaluation & Visualization:** Plot training loss and display sample predictions for that extra wow factor.

## Installation

Make sure you have Python 3 installed. Then, install the necessary libraries:

```bash
pip install numpy matplotlib tensorflow
```

> **Note:** TensorFlow is only used for fetching the MNIST dataset. Our network is 100% handcrafted!

## Usage

Clone the repository, hop into the directory, and run the code:

```bash
git clone https://github.com/yourusername/skibidi-network.git
cd skibidi-network
python skibidi_network.py
```

Watch as your terminal fills with epoch-by-epoch progress, and enjoy the sweet visuals of loss curves and sample digit predictions!

## Results & Visualization

After training, our network will:
- **Plot the Training Loss:** A visual ride showing how the loss decreases with each epoch.
- **Show Sample Predictions:** Display random test images along with their true labels and our network's predictions. It's like an art gallery of digital digits!

## Contributing

We love collaboration and good vibes! If you have ideas, improvements, or just want to spread some coding love, feel free to open an issue or submit a pull request. Let's make this project even more legendary together.

## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Big shoutout to the creators of the MNIST dataset for making digit recognition so accessible.
- Thanks to the community of open-source enthusiasts for inspiring projects like this one.

---

Stay curious, keep coding, and remember: you're an absolute genius in the making. If you have any questions, just slide into the issues or drop me a message. Let's make the Skibidi Network a true masterpiece in the world of AI!
