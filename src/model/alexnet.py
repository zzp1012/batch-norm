import torch
import torch.nn as nn

'''
modified to fit dataset size
'''
NUM_CLASSES = 10


class AlexNet_v1(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet_v1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.before_bn = nn.Sequential(
            nn.Linear(256 * 2 * 2, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
        )
        self.bn = nn.BatchNorm1d(20, eps=1e-5)
        self.after_bn = nn.Sequential(
            nn.Linear(20, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.before_bn(x)
        x = self.bn(x)
        x = self.after_bn(x)
        return x


class AlexNet_v2(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet_v2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.before_bn = nn.Sequential(
            nn.Linear(256 * 2 * 2, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
        )
        self.bn = nn.BatchNorm1d(20, eps=1e-5)
        self.after_bn = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.before_bn(x)
        x = self.bn(x)
        x = self.after_bn(x)
        return x


class AlexNet_v3(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet_v3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.before_bn = nn.Sequential(
            nn.Linear(256 * 2 * 2, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
        )
        self.bn = nn.BatchNorm1d(20, eps=1e-5)
        self.after_bn = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.before_bn(x)
        x = self.bn(x)
        x = self.after_bn(x)
        return x


class AlexNet_v4(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet_v4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.before_bn = nn.Sequential(
            nn.Linear(256 * 2 * 2, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
        )
        self.bn = nn.BatchNorm1d(20, eps=1e-5)
        self.after_bn = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.before_bn(x)
        x = self.bn(x)
        x = self.after_bn(x)
        return x