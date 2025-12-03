# XOR Classification with PyTorch

A simple feedforward neural network that learns to classify the classic XOR truth table using PyTorch. This project demonstrates how non-linear problems can be solved using multilayer neural networks.

## 🔍 Overview

The XOR function is a well-known benchmark problem in machine learning because it is not linearly separable. A neural network with at least one hidden layer is required to learn it.

This project builds and trains a neural network that takes two binary inputs and predicts whether the XOR output should be 0 or 1.
Training is performed using CrossEntropyLoss, which internally applies the softmax operation to logits.

After training, the model successfully predicts all XOR outcomes.

## 🧠 XOR Truth Table
| Input A | Input B | XOR Output |
| ------: | ------: | ---------: |
|       0 |       0 |          0 |
|       0 |       1 |          1 |
|       1 |       0 |          1 |
|       1 |       1 |          0 |

The neural network used in this project consists of:

- Input layer: 2 neurons
- Hidden layer 1: 50 neurons (ReLU activation)
- Hidden layer 2: 25 neurons (ReLU activation)
- Hidden layer 3: 10 neurons (ReLU activation)
- Output layer: 2 logits (class 0 and class 1)

Loss function: CrossEntropyLoss
- Optimizer: Adam (learning rate = 0.015)
- Number of epochs: 50

The output layer returns raw logits; CrossEntropyLoss converts them to probabilities internally.

## 🧪 Training Outcome

After training, the neural network consistently predicts the correct XOR values. The final predictions match the true XOR outputs:

Predicted: [0, 1, 1, 0]
Actual:    [0, 1, 1, 0]
