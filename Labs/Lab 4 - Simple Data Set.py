# Lab 4 - Simple Data Set

import torch
from torch.utils.data import Dataset
torch.manual_seed(1)

# Define class for dataset

class toy_set(Dataset):
    
    # Constructor with defult values 
    def __init__(self, length = 100, transform = None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform
     
    # Getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)     
        return sample
    
    # Get Length
    def __len__(self):
        return self.len


# Practice: Create a new object with length 50, and print the length of object out.
my_dataset = toy_set(length = 50)
print("My toy_set length: ", len(my_dataset))

# Practice: Construct your own my_add_mult transform. Apply my_add_mult on a new toy_set object. Print out the first three elements from the transformed dataset.

class my_add_mult(object):   
    def __init__(self, add = 2, mul = 10):
        self.add=add
        self.mul=mul
        
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.add
        y = y + self.add
        x = x * self.mul
        y = y * self.mul
        sample = x, y
        return sample
        
       
my_dataset = toy_set(transform = my_add_mult())
for i in range(3):
    x_, y_ = my_dataset[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)


from torchvision import transforms

# Practice: Make a compose as mult() execute first and then add_mult(). Apply the compose on toy_set dataset. Print out the first 3 elements in the transformed dataset.

my_compose = transforms.Compose([mult(), add_mult()])
my_transformed_dataset = toy_set(transform = my_compose)
for i in range(3):
    x_, y_ = my_transformed_dataset[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)































