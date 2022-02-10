import torch

a = torch.Tensor([1, 2, 3]).requires_grad_(True)
b = torch.Tensor([4, 5, 6]).requires_grad_(True)
c = torch.Tensor([7, 8, 9]).requires_grad_(True)

input = torch.stack([a, b, c]).view(3, 3, 1, 1)
print("batch mean: ", input.mean(dim=0).flatten())
print("batch std: ", input.std(0, unbiased=False).flatten())

bn_layer = torch.nn.BatchNorm2d(3, eps=0)
# change the running mean and running var of the BN layer to batch mean and var
bn_layer.running_mean.data.copy_(input.mean(dim=0).flatten())
bn_layer.running_var.data.copy_(input.var(0, unbiased=False).flatten())
print("running mean: ", bn_layer.running_mean)
print("running var", bn_layer.running_var)
print("gamma: ", bn_layer.weight)
print("beta: ", bn_layer.bias)

bn_layer.eval() # bn_layer.train()
output = bn_layer(input)
print("y_0: ", output[0].flatten())
print("y_1: ", output[1].flatten())
print("y_2: ", output[2].flatten())

output[0, 0, 0, 0].backward()
print("sample 1: grad", a.grad)
print("sample 2: grad", b.grad)
print("sample 3: grad", c.grad)
