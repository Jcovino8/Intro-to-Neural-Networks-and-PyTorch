# Lab 17 - Logistic Regression Prediction

import torch.nn as nn
import torch
import matplotlib.pyplot as plt 

# Set the random seed

torch.manual_seed(2)

z = torch.arange(-100, 100, 0.1).view(-1, 1)
print("The tensor: ", z)

# Create sigmoid object

sig = nn.Sigmoid()

# Use sigmoid object to calculate the 

yhat = sig(z)

yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())

# Create x and X tensor

x = torch.tensor([[1.0]])
X = torch.tensor([[1.0], [100]])
print('x = ', x)
print('X = ', X)
# Use sequential function to create model

model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

# The prediction for X

yhat = model(X)
yhat

# Create and print samples

x = torch.tensor([[1.0, 1.0]])
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
print('x = ', x)
print('X = ', X)


# Practice: Make your model and make the prediction

X = torch.tensor([-10.0])

my_model = nn.Sequential(nn.Linear(1, 1),nn.Sigmoid())
yhat = my_model(X)
print("The prediction: ", yhat)
































