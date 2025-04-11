
This is a custom-built neural network that detects handwritten numbers from image inputs. It uses ReLU in the hidden layers and a softmax activation function in the output layer for classification. The model is trained using backpropagation with a loss function to minimize prediction errors.

After training the model for 99+ epochs, the custom neural network achieved a 93% classification accuracy, demonstrating strong generalization performance on the MNIST dataset.


![image](https://github.com/user-attachments/assets/91184897-456f-499a-a45c-8b708916a055)

We start by preprocessing the input image, converting it to grayscale to reduce complexity.

![image](https://github.com/user-attachments/assets/39bb7102-4218-44a3-8a01-4cac3ac752be)

Then apply Sigmoid and Softmax activation functions during forward propagation. Based on the network’s learned weights and neuron structure, the model makes a final prediction of the digit.
![image](https://github.com/user-attachments/assets/e708d108-f627-46a6-9501-d61368c14390)

