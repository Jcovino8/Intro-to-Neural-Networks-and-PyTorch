# Lab 7 - Linear Regression 1D: Prediction

import torch

x = torch.tensor([[1.0], [2.0], [3.0]])
yhat = forward(x)
print("The prediction: ", yhat)


# Import Class Linear

from torch.nn import Linear

# Set random seed

torch.manual_seed(1)

x = torch.tensor([[1.0],[2.0],[3.0]])

x=torch.tensor([[1.0],[2.0],[3.0]])
yhat = lr(x)
print("The prediction: ", yhat)

from torch import nn

# Customize Linear Regression Class

class LR(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        
        # Inherit from parent
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    # Prediction function
    def forward(self, x):
        out = self.linear(x)
        return out

# Practice: Use the LR class to create a model and make a prediction of the following tensor.

x = torch.tensor([[1.0], [2.0], [3.0]])
x=torch.tensor([[1.0],[2.0],[3.0]])
lr1=LR(1,1)
yhat=lr1(x)
yhat



























