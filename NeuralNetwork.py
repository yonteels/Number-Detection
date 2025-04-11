import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os

# Load and preprocess MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test  = X_test.reshape(-1, 784) / 255.0
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Initialize parameters 
input_size = 784
hidden_size = 128 
output_size = 10

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)  
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size) 
b2 = np.zeros((1, output_size))

# relu function
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# Forward propagation
def forward(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

#  Loss function
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

# Backward propagation
def backward(X, y_true, z1, a1, z2, a2):
    m = X.shape[0]

    dz2 = a2 - y_true
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * (a1 > 0)  # Derivative of ReLU
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

#Training loop with mini-batch gradient descent
learning_rate = 0.01
epochs = 100
batch_size = 128
decay_rate = 0.01

for epoch in range(epochs):
    indices = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        
        z1, a1, z2, a2 = forward(X_batch)
        loss = compute_loss(y_batch, a2)
        dW1, db1, dW2, db2 = backward(X_batch, y_batch, z1, a1, z2, a2)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    learning_rate *= (1. / (1. + decay_rate * epoch))

    if epoch % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

# Evaluation
_, _, _, y_pred = forward(X_test)
accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model weights
np.savez("model_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)

# Save test images with predictions (for testing)
# output_dir = "predicted_images"
# os.makedirs(output_dir, exist_ok=True)

# N = 100  # Number of images to save
# for i in range(N):
#     img_array = (X_test[i].reshape(28, 28) * 255).astype(np.uint8)
#     img = Image.fromarray(img_array, mode='L')

#     predicted_label = np.argmax(y_pred[i])
#     actual_label = np.argmax(y_test[i])
    
#     filename = f"img_{i}_pred_{predicted_label}_true_{actual_label}.png"
#     img.save(os.path.join(output_dir, filename))

# print(f"Saved {N} prediction images in '{output_dir}' folder.")
