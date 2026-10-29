# Contains source code of the components of your model. Each component must be implemented as a class or a function in PyTorch 
# CNN
import torch.nn as nn
import torch

# baseline model
class BasicCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()

        self.block = nn.Sequential(
            # depthwise convolution
            nn.Conv2d(
                channels,
                channels,
                kernel_size=7,
                padding=3,
                groups=channels,
            ),

            # normalise channels
            nn.GroupNorm(1, channels),

            # channel expansion and projection
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels * expansion,
                channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return x + self.block(x)


# ConvNeXt model
class ConvNeXt(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super().__init__()

        self.stem = nn.Sequential(
            # input MRI image: 1 channel
            nn.Conv2d(1, 32, kernel_size=4, stride=4),
            nn.GroupNorm(1, 32),
        )

        self.features = nn.Sequential(
            # stage 1: 32 channels
            ConvNeXtBlock(32, expansion=2),

            # downsample: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.GroupNorm(1, 64),

            # stage 2: 64 channels
            ConvNeXtBlock(64, expansion=2),

            # downsample: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.GroupNorm(1, 128),

            # stage 3: 128 channels
            ConvNeXtBlock(128, expansion=2),

            # downsample: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.GroupNorm(1, 256),

            # stage 4: 256 channels
            ConvNeXtBlock(256, expansion=2),

            # global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)