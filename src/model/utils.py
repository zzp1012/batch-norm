import torch
import torch.nn as nn

class BatchNorm1d(nn.Module):

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.05,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.isDetach = True
        
        self.weight = torch.nn.Parameter(torch.ones((num_features)))
        self.bias = torch.nn.Parameter(torch.zeros((num_features)))
        self.register_buffer('running_mean', torch.zeros((num_features)))
        self.register_buffer('running_var', torch.ones((num_features)))
        self.register_buffer('num_batches_tracked', torch.tensor(0))

    def forward(self,
                input: torch.Tensor)-> torch.Tensor:
        assert len(input.shape) == 2, \
            f"the input shape is {input.shape} instead of (N, C)"
        if self.training:
            batch_mean = input.mean(0, keepdim = True) # with shape (1, C)
            batch_var = (input ** 2).mean(0, keepdim = True) # with shape (1, C)
            if self.num_batches_tracked == 0:
                self.running_mean.copy_(batch_mean.data.flatten())
                self.running_var.copy_(batch_var.data.flatten())
            else:
                self.running_mean.mul_(1 - self.momentum)
                self.running_mean.add_(self.momentum * batch_mean.data.flatten())
                self.running_var.mul_(1 - self.momentum)
                self.running_var.add_(self.momentum * batch_var.data.flatten())
            self.num_batches_tracked += 1
            if self.isDetach:
                mean_bn = torch.autograd.Variable(self.running_mean).reshape(1, self.num_features)
                var_bn = torch.autograd.Variable(self.running_var).reshape(1, self.num_features)
            else:
                mean_bn = batch_mean
                var_bn = batch_var
        else:
            mean_bn = torch.autograd.Variable(self.running_mean).reshape(1, self.num_features)
            var_bn = torch.autograd.Variable(self.running_var).reshape(1, self.num_features)
        var_bn = var_bn - mean_bn ** 2
        
        # standardization
        weight = self.weight.reshape(1, self.num_features)
        bias = self.bias.reshape(1, self.num_features)
        input_normalized = (input - mean_bn) / torch.sqrt(var_bn + self.eps)
        output = weight * input_normalized + bias
        return output