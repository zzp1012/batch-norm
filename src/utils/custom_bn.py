import torch
import torch.nn as nn

class BatchNorm1d(nn.Module):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device=None,
        dtype=None
        ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        super(BatchNorm1d,self).__init__()
        self.weight = torch.nn.Parameter(torch.ones((num_features)))
        self.bias = torch.nn.Parameter(torch.zeros((num_features)))
        self.register_buffer('running_mean',torch.zeros((num_features)))
        self.register_buffer('running_var',torch.ones((num_features)))
        self.register_buffer('num_batches_tracked',torch.tensor(0))

    def forward(
        self,
        input: torch.Tensor,
        )-> torch.Tensor:

        if self.num_batches_tracked == 0:
            batch_mean = input.mean((0), keepdim=True).detach()
            batch_var = input.var((0), unbiased=False, keepdim=True).detach()
            self.running_mean.copy_(batch_mean.data.flatten())
            self.running_var.copy_(batch_var.data.flatten())
            self.num_batches_tracked += 1
            print(batch_mean)
            print(batch_var)
        
        mean_bn = torch.autograd.Variable(self.running_mean).reshape(1,self.num_features)
        var_bn = torch.autograd.Variable(self.running_var).reshape(1,self.num_features)

        weight = self.weight.reshape(1,self.num_features)
        bias = self.bias.reshape(1,self.num_features)
        input_normalized = (input - mean_bn) / torch.sqrt(var_bn + self.eps)
        output = weight * input_normalized + bias
        return output