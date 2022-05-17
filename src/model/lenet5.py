import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5_v1(nn.Module):

    def __init__(self):
        super(LeNet5_v1, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_part = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # an affine operation: y = Wx + b
        self.before_bn = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 6*6 from image dimension
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(20)
        self.after_bn = nn.Sequential(
            nn.Linear(20, 10),
        )

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv_part(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.before_bn(x)
        x = self.bn(x)
        x = self.after_bn(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LeNet5_v2(nn.Module):

    def __init__(self):
        super(LeNet5_v2, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_part = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # an affine operation: y = Wx + b
        self.before_bn = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 6*6 from image dimension
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(20)
        self.after_bn = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv_part(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.before_bn(x)
        x = self.bn(x)
        x = self.after_bn(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LeNet5_v3(nn.Module):

    def __init__(self):
        super(LeNet5_v3, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_part = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # an affine operation: y = Wx + b
        self.before_bn = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 6*6 from image dimension
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(20)
        self.after_bn = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv_part(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.before_bn(x)
        x = self.bn(x)
        x = self.after_bn(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LeNet5_v4(nn.Module):

    def __init__(self):
        super(LeNet5_v4, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_part = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # an affine operation: y = Wx + b
        self.before_bn = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 6*6 from image dimension
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(20)
        self.after_bn = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv_part(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.before_bn(x)
        x = self.bn(x)
        x = self.after_bn(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features