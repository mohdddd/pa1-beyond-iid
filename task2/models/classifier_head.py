"""Seven-class linear classifier head on the 512-d feature, and the Net wrapper
returning (feature, logits) so every method sees the same two tensors."""
import torch.nn as nn

from task2.models.backbone import ResNet18Backbone


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int = 512, num_classes: int = 7):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, f):
        return self.fc(f)


class Net(nn.Module):
    """backbone F (ResNet-18 -> 512-d) + head C (512 -> 7)."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained)
        self.head = ClassifierHead(self.backbone.out_dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        return f, self.head(f)
