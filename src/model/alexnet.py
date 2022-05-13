import torch
import torch.nn as nn

'''
modified to fit dataset size
'''
NUM_CLASSES = 1


class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
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
        )
        self.bn = nn.BatchNorm1d(1024, eps=1e-5)
        self.after_bn = nn.Sequential(
            # nn.Dropout(inplace=False),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=False),
            # nn.Dropout(inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.before_bn(x)
        x = self.bn(x)
        x = self.after_bn(x)
        return x

if __name__ == "__main__":
    model = AlexNet()
    x = torch.rand(1, 3, 32, 32)
    print(model(x))