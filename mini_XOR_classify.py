import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim

#features
inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0,1.0]])

#labels (0 or 1)
targets = torch.tensor([0, 1, 1, 0]) #index of the correct output

# define a simple feedforward neural network for classification
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 50) #2 inputs/features, 50 neurons in hidden layer
        self.fc2 = nn.Linear(50, 25) #25 neurons fully connected to previousl layer
        self.fc3 = nn.Linear(25, 10) #10 neurons
        self.fc4 = nn.Linear(10, 2) # 2 output logits (for binary classification)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x) #weights reused here
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)  # no softmax here; CrossEntropyLoss applies it internally
        return x #output as logits

# create an instance of the network  
model = SimpleNN()
#print(model)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()# crossEntropyLoss for classification
optimizer = optim.Adam(model.parameters(), lr = 0.015)

#define epochs
epochs = 50
for e in range(epochs):
    #Zero the gradients
    optimizer.zero_grad()

    #Do a forward pass to compute the model output
    outputs = model(inputs) #output are logits

    #Compute the loss
    loss = criterion(outputs, targets)# CrossEntropyLoss compares logits to class labels

    #Backprop
    loss.backward()

    #update model's parameters (i.e. weights and biases)
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

#test model after training
test_inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0,1.0]])
test_targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)

#create predctions
with torch.no_grad():
    #forward pass on new data using test inputs
    outputs = model(test_inputs) #This calls the forward() method of your network with the input test_inputs
    #convert logits to argmax
    predicted_class = torch.argmax(outputs,dim=1) #dim=1 means: choose the index of the highest logit in each row
    print(f"Predicted Class Labels: {predicted_class}")
    print(f"Actual class labels:{test_targets}")



