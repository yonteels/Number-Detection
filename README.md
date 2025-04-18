# Overall
This is a custom-built neural network that detects handwritten numbers from image inputs. It uses ReLU activation in the hidden layers and a softmax activation function in the output layer for classification. The model is trained using backpropagation with a loss function to minimize prediction errors, achieving over 99% accuracy when predicting digits.

## Model Loss and Accuracy
After training the model for 100 epochs, the custom neural network achieved a 93% classification accuracy, demonstrating strong generalization performance on the MNIST dataset.

![image](https://github.com/user-attachments/assets/91184897-456f-499a-a45c-8b708916a055)

## Image Processing
The process begins by preprocessing the input image, including resizing and normalizing the pixel values to match the training data format.

![image](https://github.com/user-attachments/assets/39bb7102-4218-44a3-8a01-4cac3ac752be)

## Digit Prediction

During forward propagation, the model applies the ReLU activation function in the hidden layer and Softmax in the output layer. Using the learned weights and network structure, the model then predicts the most likely digit class with over 99% accuracy.

![image](https://github.com/user-attachments/assets/8e082818-b156-4ef9-8b46-e3e98840439f)

