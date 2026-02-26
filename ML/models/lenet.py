"""
LeNet-5 architecture
Original paper: "Gradient-Based Learning Applied to Document Recognition" (LeCun et al., 1998)
"""

import torch
import torch.nn as nn


class LeNet(nn.Module):
    """LeNet-5 architecture for image classification"""

    def __init__(self, num_classes=10, input_channels=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # After conv1+pool: 32->16, after conv2: 16->12, after pool2: 12->6
        # So flattened size is 16 * 6 * 6 = 576
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x