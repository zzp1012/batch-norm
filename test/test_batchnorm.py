import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/'))
from model.utils import BatchNorm1d

a = torch.Tensor([1, 2, 3]).requires_grad_(True)
b = torch.Tensor([4, 5, 6]).requires_grad_(True)
c = torch.Tensor([7, 8, 9]).requires_grad_(True)

bn_layer = BatchNorm1d(3, eps=0) # BatchNorm2d(3, eps=0)
print("running mean: ", bn_layer.running_mean)
print("running var", bn_layer.running_var)
print("gamma: ", bn_layer.weight)
print("beta: ", bn_layer.bias)

bn_layer.train()
for i in range(10):
    input = torch.stack([a, b, c])
    print("batch mean: ", input.mean(dim=0).flatten())
    print("batch std: ", input.std(0, unbiased=False).flatten())

    output = bn_layer(input)
    print("y_0: ", output[0].flatten())
    print("y_1: ", output[1].flatten())
    print("y_2: ", output[2].flatten())  

    output.sum().backward()
    print("sample 1: grad", a.grad)
    print("sample 2: grad", b.grad)
    print("sample 3: grad", c.grad)
    a.grad.zero_()
    b.grad.zero_()
    c.grad.zero_()