"""ResNet-18 feature extractor (manual §2: torchvision ResNet18_Weights.IMAGENET1K_V1,
fine-tune the complete network) and the shared frozen-BatchNorm policy (Tasks 2 & 3).

The ImageNet classifier is replaced by Identity, so the backbone outputs the
512-d global-average-pooled feature "immediately before the classifier head",
which is where MMD / the domain discriminators are attached.
"""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

FEAT_DIM = 512


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Identity()
        self.net = net
        self.out_dim = FEAT_DIM

    def forward(self, x):
        return self.net(x)          # (B, 512)


def freeze_bn_stats(module: nn.Module) -> int:
    """Manual BN policy: after model.train(), put ONLY BatchNorm modules in eval
    mode so running mean/var stay at their ImageNet values. gamma/beta remain
    trainable (requires_grad is untouched). Returns the number of BN modules."""
    n = 0
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            n += 1
    return n
