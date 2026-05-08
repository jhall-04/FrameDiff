"""
Simple CNN for frame difference classification.
"""
from torch import nn
import torch

class FrameDiffModel(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            # Build initial feature map extracting information from the 2-channel input (frame difference)  
            nn.Conv2d(in_channels, 16, 3, padding=1),
            # Normalize over the batch to stabilize training and improve convergence
            nn.BatchNorm2d(16),
            # Introduce non-linearity to allow the model to learn complex patterns in the data
            nn.ReLU(inplace=True),
            # Reduce spatial dimensions while retaining important features, making the model more efficient and less prone to overfitting
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze(-1)
    