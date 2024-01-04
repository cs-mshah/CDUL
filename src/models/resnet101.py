from torch import nn
from torchvision import models

class ResNet101(nn.Module):

    def __init__(self, num_labels):
        super(ResNet101, self).__init__()
        self.net = models.resnet101(weights='DEFAULT')
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_labels)

    def forward(self, x):
        x = self.net(x)
        return x