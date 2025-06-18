# Lab 2 - 2D Tensors

# These are the libraries will be used for this lab.

import numpy as np 
import matplotlib.pyplot as plt
import torch
import pandas as pd

# Practice: try to convert Pandas Series to tensor

df = pd.DataFrame({'A':[11, 33, 22],'B':[3, 3, 2]})
converted_tensor = torch.tensor(df.values)
print ("Tensor: ", converted_tensor)

# Give an idea on tensor_obj[number: number][number]

tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
sliced_tensor_example = tensor_example[1:3]
print("1. Slicing step on tensor_example: ")
print("Result after tensor_example[1:3]: ", sliced_tensor_example)
print("Dimension after tensor_example[1:3]: ", sliced_tensor_example.ndimension())
print("================================================")
print("2. Pick an index on sliced_tensor_example: ")
print("Result after sliced_tensor_example[1]: ", sliced_tensor_example[1])
print("Dimension after sliced_tensor_example[1]: ", sliced_tensor_example[1].ndimension())
print("================================================")
print("3. Combine these step together:")
print("Result: ", tensor_example[1:3][1])
print("Dimension: ", tensor_example[1:3][1].ndimension())

# Practice: Use slice and index to change the values on the matrix tensor_ques.

tensor_ques = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
tensor_ques[1:3, 1] = 0
print("The result: ", tensor_ques)

# Practice: Calculate the product of two tensors (X and Y) with different sizes 

X = torch.tensor([[0, 1], [1, 2]])
Y = torch.tensor([[-1, -2, 0], [2, 1, 2]])
X_times_Y = torch.mm(X, Y)
print("The result of X * Y: ", X_times_Y)






















