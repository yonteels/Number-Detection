#python predict_digit.py my_digit.png

import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt

# Load model weights
weights = np.load("model_weights.npz")
W1 = weights['W1']
b1 = weights['b1']
W2 = weights['W2']
b2 = weights['b2']

# Activation functions
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# Forward pass (weight adjustment, sigmoid, softmax, weight adjustment )
def forward(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return a2

# Preprocess image
def preprocess_image(path):
    img = Image.open(path).convert("L")     
    img = img.resize((28, 28))            
    img = np.array(img)

    plt.imshow(img, cmap='gray')
    plt.title("Original Image (Grayscale)")
    plt.show()

    # Normalize image by dividing by 255.0 
    img = img / 255.0

    # Invert the image if digit is dark 
    # img = 1.0 - img

    plt.imshow(img, cmap='gray')
    plt.title("Processed Image (Normalized)")
    plt.show()

    img = img.reshape(1, 784) 
    return img

# Main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_digit.py path_to_image")
        sys.exit(1)

    image_path = sys.argv[1]
    X = preprocess_image(image_path)
    prediction = forward(X)

    print("Prediction Probabilities:", prediction)

    predicted_digit = np.argmax(prediction)
    print(f"Predicted digit: {predicted_digit}")

    bar_x = np.array([0,1,2,3,4,5,6,7,8,9])
    bar_y = bar_y = prediction.flatten()

    plt.bar(bar_x, bar_y)
    plt.show()