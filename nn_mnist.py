import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Define our tanh activation and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

# Load and preprocess MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0  # Flatten and normalize
X_test = X_test.reshape(-1, 784) / 255.0
y_train = to_categorical(y_train, 10)       # One-hot encode labels
y_test = to_categorical(y_test, 10)

# Network architecture parameters
input_dim = 784          
hidden_dim = 64         
output_dim = 10          # 10 output neurons for digits 0-9

# Initialize weights and biases (with our signature flair)
W1 = np.random.uniform(-0.5, 0.5, (input_dim, hidden_dim))
b1 = np.full((1, hidden_dim), 0.5)
W2 = np.random.uniform(-0.5, 0.5, (hidden_dim, output_dim))
b2 = np.full((1, output_dim), 0.7)

# Training hyperparameters
learning_rate = 0.01
epochs = 10
batch_size = 64
num_batches = X_train.shape[0] // batch_size

loss_history = []  # For tracking training loss over epochs

# Training loop with mini-batch gradient descent
for epoch in range(epochs):
    # Shuffle the data for a fresh vibe each epoch
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]
    
    epoch_loss = 0
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]
        
        # Forward pass
        Z1 = np.dot(X_batch, W1) + b1
        A1 = tanh(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = tanh(Z2)
        
        # Mean Squared Error loss calculation
        loss = np.mean((A2 - y_batch) ** 2)
        epoch_loss += loss
        
        # Backpropagation for gradients
        dA2 = (A2 - y_batch)
        dZ2 = dA2 * tanh_deriv(Z2)
        dW2 = np.dot(A1.T, dZ2) / batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size
        
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * tanh_deriv(Z1)
        dW1 = np.dot(X_batch.T, dZ1) / batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size
        
        # Update our parameters with gradient descent magic
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        
    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluate the network on the test set
Z1_test = np.dot(X_test, W1) + b1
A1_test = tanh(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = tanh(Z2_test)

# Get predictions by taking the argmax of the outputs
predictions = np.argmax(A2_test, axis=1)
true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == true_labels)
print("Test accuracy:", accuracy)

# Visualize the training loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), loss_history, marker='o', color='purple', label='Training Loss')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Visualize some sample predictions from the test set
num_samples = 10
sample_indices = np.random.choice(range(X_test.shape[0]), num_samples, replace=False)

plt.figure(figsize=(15, 4))
for i, idx in enumerate(sample_indices):
    image = X_test[idx].reshape(28, 28)
    pred_label = predictions[idx]
    true_label = true_labels[idx]
    plt.subplot(1, num_samples, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()
