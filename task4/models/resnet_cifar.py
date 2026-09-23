"""CIFAR ResNet-18 (manual §4): torchvision ResNet-18 with 7x7/2 conv1 -> 3x3/1, max-pool removed,
random initialisation, 32x32 inputs. Exposes the layer2/layer3 split used by PROSER manifold mixup."""
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet


class CifarResNet18(ResNet):
    feat_dim = 512

    def __init__(self, num_classes: int = 10, num_dummy: int = 0):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")  # torchvision's conv init
        self.maxpool = nn.Identity()
        # PROSER dummy classifiers (classifier placeholders); absent for Vanilla / GCSC
        self.dummy = nn.Linear(self.feat_dim, num_dummy) if num_dummy else None

    def pre(self, x):                       # input -> end of layer2
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer2(self.layer1(x))

    def post(self, h):                      # layer3 -> penultimate feature f(x), 512-d
        return torch.flatten(self.avgpool(self.layer4(self.layer3(h))), 1)

    def features(self, x):
        return self.post(self.pre(x))

    def forward(self, x):                   # known-class logits z(x)
        return self.fc(self.features(x))
