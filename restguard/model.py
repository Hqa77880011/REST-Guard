import torch
from torch import nn


class EEGNet(nn.Module):
    def __init__(self, samples):
        super().__init__()
        self.temporal = nn.Sequential(nn.Conv2d(1, 8, (1, 64), padding="same", bias=False), nn.BatchNorm2d(8))
        self.spatial = nn.Conv2d(8, 16, (3, 1), groups=8, bias=False)
        self.first = nn.Sequential(nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.25))
        self.second = nn.Sequential(nn.Conv2d(16, 16, (1, 16), groups=16, padding="same", bias=False),
                                    nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ELU(),
                                    nn.AvgPool2d((1, 8)), nn.Dropout(0.25), nn.Flatten())
        self.classifier = nn.Linear(16 * (samples // 32), 2)

    def forward(self, x):
        encoded = self.temporal(x[:, None])
        return self.classifier(self.second(self.first(self.spatial(encoded))))

    @torch.no_grad()
    def constrain(self):
        for weight, maximum in ((self.spatial.weight, 1.0), (self.classifier.weight, 0.25)):
            norm = weight.flatten(1).norm(dim=1).clamp_min(1e-8)
            weight.mul_((maximum / norm).clamp_max(1).view(-1, *([1] * (weight.ndim - 1))))
