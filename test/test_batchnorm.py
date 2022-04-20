import os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/'))
from model.utils import BatchNorm2d

a = torch.Tensor([1, 2, 3]).requires_grad_(True)
b = torch.Tensor([4, 5, 6]).requires_grad_(True)
c = torch.Tensor([7, 8, 9]).requires_grad_(True)
d = torch.Tensor([10, 11, 12]).requires_grad_(True)
e = torch.Tensor([13, 14, 15]).requires_grad_(True)

bn_layer = nn.BatchNorm2d(3, eps=0) # BatchNorm2d(3, eps=0)
print("running mean: ", bn_layer.running_mean)
print("running var", bn_layer.running_var)
print("gamma: ", bn_layer.weight)
print("beta: ", bn_layer.bias)

input = torch.stack([d, e, c]).view(3, 3, 1, 1)
print("batch mean: ", input.mean(dim=0).flatten())
print("batch std: ", input.std(0, unbiased=False).flatten())

bn_layer.train() # bn_layer.train()
output = bn_layer(input)
print("y_0: ", output[0].flatten())
print("y_1: ", output[1].flatten())
print("y_2: ", output[2].flatten())

output.sum().backward()
print("sample 1: grad", d.grad)
print("sample 2: grad", e.grad)
print("sample 3: grad", c.grad)

a.grad = None
b.grad = None
c.grad = None

bn_layer = BatchNorm2d(3, eps=0) # BatchNorm2d(3, eps=0)
print("running mean: ", bn_layer.running_mean)
print("running var", bn_layer.running_var)
print("gamma: ", bn_layer.weight)
print("beta: ", bn_layer.bias)

input = torch.stack([d, e, c]).view(3, 3, 1, 1)
print("batch mean: ", input.mean(dim=0).flatten())
print("batch std: ", input.std(0, unbiased=False).flatten())

bn_layer.train() # bn_layer.train()
output = bn_layer(input)
print("y_0: ", output[0].flatten())
print("y_1: ", output[1].flatten())
print("y_2: ", output[2].flatten())

# output[:, 0].sum().backward()
output.sum().backward()
print("sample 1: grad", d.grad)
print("sample 2: grad", e.grad)
print("sample 3: grad", c.grad)