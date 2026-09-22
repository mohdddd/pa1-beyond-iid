"""Task 3 uses exactly the Task 2 ResNet-18 backbone and frozen-BN policy (manual §3:
same initialisation / preprocessing). Re-exported so the shared definition cannot drift."""
from task2.models.backbone import FEAT_DIM, ResNet18Backbone, freeze_bn_stats  # noqa: F401
